import math
import numpy as np
from module.basic_fn import *
from module.mediapipe_op import KEY_LANDMARKS
from module.upper_body import *
from module.lower_body import *

def cal_angle_3point(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# =========================================================
# 정적 자세 측정 계산
# =========================================================

def calculate_horizontal_tilt_pixel(point_a, point_b, width, height):
    """두 점을 잇는 선의 수평축 기준 기울기를 도 단위로 계산한다."""
    ax = float(point_a[0] * width)
    ay = float(point_a[1] * height)
    bx = float(point_b[0] * width)
    by = float(point_b[1] * height)

    dx = bx - ax
    dy = -(by - ay)  # OpenCV는 아래쪽이 +y이므로 수학 좌표계에 맞게 반전

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    angle = math.degrees(math.atan2(dy, dx))

    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0

    return float(angle)


def calculate_knee_alignment(hip, knee, ankle, width, height):
    """골반-발목 기준선에서 무릎이 벗어난 정도를 다리 길이 대비 %로 반환한다."""
    hip_xy = np.array([hip[0] * width, hip[1] * height], dtype=np.float32)
    knee_xy = np.array([knee[0] * width, knee[1] * height], dtype=np.float32)
    ankle_xy = np.array([ankle[0] * width, ankle[1] * height], dtype=np.float32)

    leg_vector = ankle_xy - hip_xy
    leg_length = float(np.linalg.norm(leg_vector))

    if leg_length < 1e-6:
        return None

    knee_vector = knee_xy - hip_xy

    # 2차원 외적의 z 성분
    cross_value = abs(
        float(
            leg_vector[0] * knee_vector[1]
            - leg_vector[1] * knee_vector[0]
        )
    )

    distance_to_line = cross_value / leg_length
    return float(distance_to_line / leg_length * 100.0)


# =========================================================
# HEAD, NECK_CENTER등의 추가 포인트 생성
# =========================================================
def create_pose_points(landmarks):
    """좌표 딕셔너리 + visibility 딕셔너리를 함께 반환."""
    points = {}
    vis = {}
    for name, index in KEY_LANDMARKS.items():
        lm = landmarks[index]
        points[name] = np.array([lm.x, lm.y], dtype=np.float32)
        vis[name] = getattr(lm, "visibility", 1.0)

    points["HEAD"] = midpoint(
        landmarks[KEY_LANDMARKS['left_ear']],
        landmarks[KEY_LANDMARKS['right_ear']]
    )
    points["NECK_CENTER"] = midpoint(
        landmarks[KEY_LANDMARKS['left_shoulder']],
        landmarks[KEY_LANDMARKS['right_shoulder']]
    )
    return points, vis

# =====================================
# Calculate Pose feature
# =====================================
def calculate_all_features(landmarks, width, height, vis_th=0.5):
    """
    각 지표를 계산하되, 필요한 관절의 visibility가 낮으면 그 지표는 None.
    vis_th: 이 값 미만이면 신뢰 불가로 간주.
    """
    points, vis = create_pose_points(landmarks)

    def ok(*names):
        """지정한 관절들이 모두 신뢰 가능한지."""
        return all(vis.get(n, 1.0) >= vis_th for n in names)

    def safe(cond, fn):
        """cond가 True일 때만 계산, 아니면 None."""
        return fn() if cond else None

    features = {
        # ===== 정면 지표 =====
        "hip_tilt_deg": safe(
            ok("left_hip", "right_hip"),
            lambda: calculate_horizontal_tilt_pixel(
                points["left_hip"], points["right_hip"], width, height),
        ),
        "left_knee_alignment": safe(
            ok("left_hip", "left_knee", "left_ankle"),
            lambda: calculate_knee_alignment(
                points["left_hip"], points["left_knee"], points["left_ankle"], width, height),
        ),
        "right_knee_alignment": safe(
            ok("right_hip", "right_knee", "right_ankle"),
            lambda: calculate_knee_alignment(
                points["right_hip"], points["right_knee"], points["right_ankle"], width, height),
        ),
        "shoulder_tilt_deg": safe(
            ok("left_shoulder", "right_shoulder"),
            lambda: calculate_shoulder_tilt(
                points["left_shoulder"], points["right_shoulder"], width, height),
        ),

        # ===== 측면 지표 =====
        "fha_deg": safe(
            ok("left_shoulder", "right_shoulder", "left_ear", "right_ear"),
            lambda: calculate_fha(
                points["NECK_CENTER"], points["HEAD"], width, height),
        ),
        "fsa_deg": safe(
            ok("left_shoulder", "right_shoulder"),
            lambda: calculate_fsa(
                points["NECK_CENTER"], points["left_shoulder"], width, height),
        ),
        "thoracic_kyphosis_deg": safe(
            ok("left_ear", "right_ear", "left_shoulder", "left_hip"),
            lambda: calculate_thoracic_kyphosis(
                points["HEAD"], points["left_shoulder"], points["left_hip"], width, height),
        ),

        # ===== 하체 지표 =====
        "pelvic_tilt_ant_deg": safe(
            ok("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"),
            lambda: calculate_pelvic_tilt_ant(
                points["left_shoulder"], points["right_shoulder"],
                points["left_hip"], points["right_hip"],
                points["left_knee"], points["right_knee"], width, height),
        ),
        "left_knee_valgus_deg": safe(
            ok("left_hip", "left_knee", "left_ankle"),
            lambda: calculate_knee_valgus_angle(
                points["left_hip"], points["left_knee"], points["left_ankle"], width, height),
        ),
        "right_knee_valgus_deg": safe(
            ok("right_hip", "right_knee", "right_ankle"),
            lambda: calculate_knee_valgus_angle(
                points["right_hip"], points["right_knee"], points["right_ankle"], width, height),
        ),
    }
    return features

# =====================================
# Classify Features
# =====================================
def classify_all_features(features):
    """
    계산된 features를 각 지표별로 정상/이상 분류.
    features: calculate_all_features 결과 딕셔너리
    """
    statuses = {
        "fha":               classify_fha(features["fha_deg"]),
        "fsa":               classify_fsa(features["fsa_deg"]),
        "shoulder_tilt":     classify_shoulder_tilt(features["shoulder_tilt_deg"]),
        "thoracic_kyphosis": classify_thoracic_kyphosis(features["thoracic_kyphosis_deg"]),
        "pelvic_tilt_ant":   classify_pelvic_tilt_ant(features["pelvic_tilt_ant_deg"]),
    }
    return statuses

def features_to_metrics(front_features, side_features):
    """
    정면/측면 features를 REF_RANGES가 기대하는 metrics 키로 변환.
    front_features: 정면 이미지에서 계산한 features
    side_features:  측면 이미지에서 계산한 features
    """
    # 무릎 정렬: 좌우 평균 (정면 기준)
    lk = front_features.get("left_knee_valgus_deg")
    rk = front_features.get("right_knee_valgus_deg")
    if lk is not None and rk is not None:
        knee = (lk + rk) / 2.0
    else:
        knee = lk if lk is not None else rk

    metrics = {
        # 정면 지표
        "shoulder_tilt":     front_features.get("shoulder_tilt_deg"),
        "knee_valgus":       knee,
        # 측면 지표
        "forward_head":      side_features.get("fha_deg"),
        "round_shoulder":    side_features.get("fsa_deg"),
        "thoracic_kyphosis": side_features.get("thoracic_kyphosis_deg"),
    }

    # None 값은 제외 (find_abnormal이 판정 못 하므로)
    return {k: v for k, v in metrics.items() if v is not None}