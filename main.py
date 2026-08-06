import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from upper_body import calculate_fha
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("models/pose_landmarker_full.task")


# =========================================================
# 0. 설정값
# =========================================================

# 프로젝트에서 사용할 MediaPipe Pose 랜드마크
# 짧은 이름을 랜드마크 번호에 바로 연결한다.
POINT_MAPPING = {
    "LS": 11,  # Left Shoulder
    "RS": 12,  # Right Shoulder
    "LE": 13,  # Left Elbow
    "RE": 14,  # Right Elbow
    "LW": 15,  # Left Wrist
    "RW": 16,  # Right Wrist
    "LH": 23,  # Left Hip
    "RH": 24,  # Right Hip
    "LK": 25,  # Left Knee
    "RK": 26,  # Right Knee
    "LA": 27,  # Left Ankle
    "RA": 28,  # Right Ankle
}

LEFT_EAR_INDEX = 7
RIGHT_EAR_INDEX = 8
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12


# 화면에 표시할 연결선
POSE_CONNECTIONS = [
    ("HEAD", "NECK_CENTER"),
    ("LS", "RS"),
    ("LH", "RH"),

    ("LS", "LE"),
    ("LE", "LW"),
    ("RS", "RE"),
    ("RE", "RW"),

    ("LS", "LH"),
    ("LH", "LK"),
    ("LK", "LA"),

    ("RS", "RH"),
    ("RH", "RK"),
    ("RK", "RA"),
]


FEATURE_DISPLAY = [
    ("Shoulder tilt", "shoulder_tilt_deg"),
    ("Hip tilt", "hip_tilt_deg"),
    ("Left knee", "left_knee_alignment"),
    ("Right knee", "right_knee_alignment"),
    ("FHA", "fha_deg"),
]


# =========================================================
# 1. 좌표 변환 및 가상점 생성
# =========================================================

def array_to_pixel(point, width, height):
    """정규화 좌표 [x, y]를 OpenCV 픽셀 좌표로 변환한다."""
    return int(point[0] * width), int(point[1] * height)


def midpoint(point_a, point_b):
    """MediaPipe Landmark 두 점의 정규화된 2차원 중간점을 계산한다."""
    return np.array(
        [
            (point_a.x + point_b.x) / 2.0,
            (point_a.y + point_b.y) / 2.0,
        ],
        dtype=np.float32,
    )


# =========================================================
# 2. 정적 자세 측정 계산
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
# 3. 핵심 포인트 생성
# =========================================================

def create_pose_points(landmarks):
    """
    측정과 디버깅에 사용할 좌표를 하나의 딕셔너리로 만든다.

    HEAD: 양쪽 귀의 중간점
    NECK_CENTER: 양쪽 어깨의 중간점
    나머지 점: MediaPipe 랜드마크의 정규화된 x, y 좌표
    """
    points = {
        name: np.array(
            [landmarks[index].x, landmarks[index].y],
            dtype=np.float32,
        )
        for name, index in POINT_MAPPING.items()
    }

    points["HEAD"] = midpoint(
        landmarks[LEFT_EAR_INDEX],
        landmarks[RIGHT_EAR_INDEX],
    )

    points["NECK_CENTER"] = midpoint(
        landmarks[LEFT_SHOULDER_INDEX],
        landmarks[RIGHT_SHOULDER_INDEX],
    )

    return points





# =========================================================
# 5. 다음 단계 전달용 결과 생성
# =========================================================

def create_pose_result(points, features, timestamp_ms):
    """2차원 좌표와 특징값을 직렬화 가능한 딕셔너리로 만든다."""
    point_result = {
        name: {
            "x": float(point[0]),
            "y": float(point[1]),
        }
        for name, point in points.items()
    }

    feature_result = {
        name: round(value, 3) if value is not None else None
        for name, value in features.items()
    }

    return {
        "timestamp_ms": timestamp_ms,
        "points": point_result,
        "features": feature_result,
    }


def create_llm_front_data(features):
    """LLM 담당자에게 전달할 정면 측정값 딕셔너리."""
    return {
        "front": {
            "shoulder_tilt": (
                round(features["shoulder_tilt_deg"], 1)
                if features["shoulder_tilt_deg"] is not None
                else None
            ),
            "hip_tilt": (
                round(features["hip_tilt_deg"], 1)
                if features["hip_tilt_deg"] is not None
                else None
            ),
            "knee_alignment": {
                "left": (
                    round(features["left_knee_alignment"], 1)
                    if features["left_knee_alignment"] is not None
                    else None
                ),
                "right": (
                    round(features["right_knee_alignment"], 1)
                    if features["right_knee_alignment"] is not None
                    else None
                ),
            },
        }
    }


# =========================================================
# 6. 디버깅 화면 출력
# =========================================================

def draw_debug_pose(frame, points, width, height):
    """필요한 점과 연결선만 간단히 화면에 표시한다."""
    pixel_points = {
        name: array_to_pixel(point, width, height)
        for name, point in points.items()
    }

    # 연결선 먼저 그려서 점이 선 위에 보이도록 한다.
    for start_name, end_name in POSE_CONNECTIONS:
        if start_name not in pixel_points or end_name not in pixel_points:
            continue

        cv2.line(
            frame,
            pixel_points[start_name],
            pixel_points[end_name],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 점만 간단히 표시한다. 이름 라벨은 생략한다.
    for name, pixel_point in pixel_points.items():
        radius = 6 if name in ("HEAD", "NECK_CENTER") else 4

        cv2.circle(
            frame,
            pixel_point,
            radius,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_feature_values(frame, features):
    """측정 결과를 디버깅 화면 왼쪽에 표시한다."""
    y_position = 30

    for label, key in FEATURE_DISPLAY:
        value = features.get(key)
        text = f"{label}: None" if value is None else f"{label}: {value:.1f}"

        cv2.putText(
            frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y_position += 28


# =========================================================
# 7. Main
# =========================================================
# file exist check 
def main():
    if not MODEL_PATH.exists():  # model exist check
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )

    options = vision.PoseLandmarkerOptions(  # pose landmarker options
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,  # camera frame sequential procces
        num_poses=1,                            # exepct one for exact judgement
        min_pose_detection_confidence=0.5,      # standard      
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    # camera check operation
    cap = cv2.VideoCapture(0)   

    if not cap.isOpened():   
        raise RuntimeError("카메라를 열 수 없습니다.")
    # video time scheduler
    start_time = time.perf_counter() # save standard time 
    previous_timestamp_ms = -1

    try:
        with vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    print("프레임을 읽지 못했습니다.")
                    break

                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                if timestamp_ms <= previous_timestamp_ms:
                    timestamp_ms = previous_timestamp_ms + 1

                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                features = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    points = create_pose_points(landmarks)

                    features = {
                        "shoulder_tilt_deg":
                            calculate_horizontal_tilt_pixel(
                                points["LS"],
                                points["RS"],
                                width,
                                height,
                            ),

                        "hip_tilt_deg":
                            calculate_horizontal_tilt_pixel(
                                points["LH"],
                                points["RH"],
                                width,
                                height,
                            ),

                        "left_knee_alignment":
                            calculate_knee_alignment(
                                points["LH"],
                                points["LK"],
                                points["LA"],
                                width,
                                height,
                            ),

                        "right_knee_alignment":
                            calculate_knee_alignment(
                                points["RH"],
                                points["RK"],
                                points["RA"],
                                width,
                                height,
                            ),
                          "fha_deg":
                            calculate_fha(
                                points["NECK_CENTER"],
                                points["HEAD"],
                                width,
                                height,
            ),    
                    }

                    draw_debug_pose(
                        frame,
                        points,
                        width,
                        height,
                    )

                    draw_feature_values(
                        frame,
                        features,
                    )

                else:
                    cv2.putText(
                        frame,
                        "Pose not detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )


               




                cv2.imshow(
                    "Static Front Posture Measurement",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("c") and features is not None:
                    llm_data = create_llm_front_data(features)

                    print(
                        json.dumps(
                            llm_data,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()