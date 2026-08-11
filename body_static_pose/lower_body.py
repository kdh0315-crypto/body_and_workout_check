import math
import numpy as np


# ========================================================
# +. Pelvic tilt Ant
# ========================================================

def midpoint2D(point_a, point_b):
    return np.array([
        (point_a[0] + point_b[0]) / 2.0,
        (point_a[1] + point_b[1]) / 2.0,
    ], dtype=np.float32)


def calculate_vector_angle(vector_a, vector_b):
    norm = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if norm < 1e-6:
        return None

    dot = np.dot(vector_a, vector_b)
    cos = np.clip(dot / norm, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos)))


def calculate_pelvic_tilt_ant(
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip,
    left_knee,
    right_knee,
    width,
    height,
):
    # normalized coordinate -> pixel coordinate
    left_shoulder_px = np.array([
        left_shoulder[0] * width,
        left_shoulder[1] * height,
    ], dtype=np.float32)

    right_shoulder_px = np.array([
        right_shoulder[0] * width,
        right_shoulder[1] * height,
    ], dtype=np.float32)

    left_hip_px = np.array([
        left_hip[0] * width,
        left_hip[1] * height,
    ], dtype=np.float32)

    right_hip_px = np.array([
        right_hip[0] * width,
        right_hip[1] * height,
    ], dtype=np.float32)

    left_knee_px = np.array([
        left_knee[0] * width,
        left_knee[1] * height,
    ], dtype=np.float32)

    right_knee_px = np.array([
        right_knee[0] * width,
        right_knee[1] * height,
    ], dtype=np.float32)

    # center points
    shoulder_center = midpoint2D(
        left_shoulder_px,
        right_shoulder_px,
    )

    hip_center = midpoint2D(
        left_hip_px,
        right_hip_px,
    )

    knee_center = midpoint2D(
        left_knee_px,
        right_knee_px,
    )

    # hip as vertex
    hip_to_shoulder = shoulder_center - hip_center
    hip_to_knee = knee_center - hip_center

    internal_angle = calculate_vector_angle(
        hip_to_shoulder,
        hip_to_knee,
    )

    if internal_angle is None:
        return None

    # straight alignment = 0 deg
    return float(180.0 - internal_angle)


def classify_pelvic_tilt_ant(angle):
    if angle is None:
        return "measurement_failed"

    if 8.0 <= angle <= 18.0:
        return "normal"

    return "abnormal"


   # ========================================================
# +. Knee alignment
# ========================================================

def calculate_knee_alignment(
    hip,
    knee,
    ankle,
    width,
    height,
    side,
):
    # normalized coordinate -> pixel coordinate
    hip_xy = np.array([
        hip[0] * width,
        hip[1] * height,
    ], dtype=np.float32)

    knee_xy = np.array([
        knee[0] * width,
        knee[1] * height,
    ], dtype=np.float32)

    ankle_xy = np.array([
        ankle[0] * width,
        ankle[1] * height,
    ], dtype=np.float32)

    # knee -> hip / ankle vector
    knee_to_hip = hip_xy - knee_xy
    knee_to_ankle = ankle_xy - knee_xy

    # 두 벡터 사이의 내부각
    internal_angle = calculate_vector_angle(
        knee_to_hip,
        knee_to_ankle,
    )

    if internal_angle is None:
        return None

    # 일직선 정렬 = 0 deg
    deviation = float(180.0 - internal_angle)

    # HKA neutral 기준 참고: 3 deg 이하 normal
    if deviation <= 3.0:
        return {
            "angle": deviation,
            "direction": "normal",
        }

    # hip-ankle 기준선에서 knee의 좌우 위치 확인
    dy = ankle_xy[1] - hip_xy[1]
    if abs(dy) < 1e-6:
        return None

    t = (knee_xy[1] - hip_xy[1]) / dy
    line_x = hip_xy[0] + t * (ankle_xy[0] - hip_xy[0])
    knee_offset = knee_xy[0] - line_x

    is_valgus = knee_offset < 0 if side == "left" else knee_offset > 0
    direction = "valgus" if is_valgus else "varus"

    return {
        "angle": deviation,
        "direction": direction,
    }
