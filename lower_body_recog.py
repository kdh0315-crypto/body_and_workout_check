import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("models/pose_landmarker_full.task")


# =========================================================
# 0. 설정값
# =========================================================

# MediaPipe Pose landmark index
LANDMARK_INDEX = {
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,

    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,

    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,

    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,

    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,

    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,

    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}


# 현재 사용하는 핵심 포인트 매핑
POINT_MAPPING = {
    "LS": "LEFT_SHOULDER",
    "RS": "RIGHT_SHOULDER",
     "LE": "LEFT_ELBOW",
    "RE": "RIGHT_ELBOW",

    "LW": "LEFT_WRIST",
    "RW": "RIGHT_WRIST",

    "LH": "LEFT_HIP",
    "RH": "RIGHT_HIP",

    "LK": "LEFT_KNEE",
    "RK": "RIGHT_KNEE",

    "LA": "LEFT_ANKLE",
    "RA": "RIGHT_ANKLE",
}


# 화면에 표시할 연결선
# (시작점, 끝점, 색상(BGR), 굵기)
POSE_CONNECTIONS = [
    ("HEAD", "NECK_CENTER", (255, 0, 255), 2),

    # 어깨와 골반
    ("LS", "RS", (0, 255, 0), 2),
    ("LH", "RH", (0, 255, 0), 2),

    # 왼팔
    ("LS", "LE", (255, 0, 0), 3),
    ("LE", "LW", (255, 0, 0), 3),

    # 오른팔
    ("RS", "RE", (0, 0, 255), 3),
    ("RE", "RW", (0, 0, 255), 3),

    # 왼쪽 몸과 다리
    ("LS", "LH", (255, 0, 0), 3),
    ("LH", "LK", (255, 0, 0), 3),
    ("LK", "LA", (255, 0, 0), 3),

    # 오른쪽 몸과 다리
    ("RS", "RH", (0, 0, 255), 3),
    ("RH", "RK", (0, 0, 255), 3),
    ("RK", "RA", (0, 0, 255), 3),
]

# 화면 왼쪽에 표시할 특징값
FEATURE_DISPLAY = [
    ("Shoulder tilt", "shoulder_tilt_deg"),
    ("Hip tilt", "hip_tilt_deg"),
    ("Left knee", "left_knee_alignment"),
    ("Right knee", "right_knee_alignment"),
]


# =========================================================
# 1. 좌표 변환 및 가상점 생성
# =========================================================

def array_to_pixel(point, width, height):
    """
    정규화 좌표 [x, y, z]를
    OpenCV 화면용 픽셀 좌표 (x, y)로 변환한다.
    """
    x = int(point[0] * width)
    y = int(point[1] * height)

    return x, y


def landmark_to_array(landmark):
    """
    MediaPipe Landmark 객체를
    [x, y, z] NumPy 배열로 변환한다.
    """
    return np.array(
        [
            landmark.x,
            landmark.y,
            landmark.z,
        ],
        dtype=np.float32,
    )


def midpoint(point_a, point_b):
    """
    MediaPipe Landmark 두 점의 3차원 중간점을 계산한다.
    """
    return np.array(
        [
            (point_a.x + point_b.x) / 2.0,
            (point_a.y + point_b.y) / 2.0,
            (point_a.z + point_b.z) / 2.0,
        ],
        dtype=np.float32,
    )


# =========================================================
# 2. 정적 자세 측정 계산
# =========================================================

def calculate_horizontal_tilt_pixel(point_a, point_b, width, height):
    """두 점을 잇는 선의 수평축 기준 기울기(도)를 계산한다."""
    ax = float(point_a[0] * width)
    ay = float(point_a[1] * height)
    bx = float(point_b[0] * width)
    by = float(point_b[1] * height)

    dx = bx - ax
    dy = -(by - ay)

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
    leg_length = np.linalg.norm(leg_vector)

    if leg_length < 1e-6:
        return None

    knee_vector = knee_xy - hip_xy
    cross_value = abs(float(np.cross(leg_vector, knee_vector)))
    distance_to_line = cross_value / leg_length

    return float(distance_to_line / leg_length * 100.0)


# =========================================================
# 3. 핵심 포인트 생성
# =========================================================

def create_pose_points(landmarks):
 
    head_center = midpoint(
        landmarks[LANDMARK_INDEX["LEFT_EAR"]],
        landmarks[LANDMARK_INDEX["RIGHT_EAR"]],
    )

    neck_center = midpoint(
        landmarks[LANDMARK_INDEX["LEFT_SHOULDER"]],
        landmarks[LANDMARK_INDEX["RIGHT_SHOULDER"]],
    )

    normalized_points = {
        name: landmark_to_array(
            landmarks[LANDMARK_INDEX[index_name]]
        )
        for name, index_name in POINT_MAPPING.items()
    }

    normalized_points["HEAD"] = head_center

    return normalized_points, neck_center

# ========================================================
# +. Pelvic tilt Ant
# ========================================================
# additional function to calculate midpoint in 2D (no Z dimension)
def midpoint2D(point_a, point_b):
    midpoint = np.array([
        (point_a.x + point_b.x) / 2.0,
        (point_a.y + point_b.y) / 2.0], dtype=np.float32)

    return midpoint

# additional function to calculate two vectors' angle 
# there is rule that vector is 'numpy array'
def cal_vector_angle(vector_a, vector_b):
    # do inner product
    dot  = np.dot(vector_a, vector_b)
    norm = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos  = np.clip(dot / norm, -1, 1) # clip to protect domain error at arccos

    return np.degrees(np.arccos(cos)) # calculate angle with arccos

def get_pelvic_tilt_ant(landmarks):
    # get center value of hip, shoulder, and knee
    hip_center = midpoint2D(landmarks[LANDMARK_INDEX["LEFT_HIP"]], landmarks[LANDMARK_INDEX["RIGHT_HIP"]])
    shoulder_center = midpoint2D(landmarks[LANDMARK_INDEX["LEFT_SHOULDER"]], landmarks[LANDMARK_INDEX["RIGHT_SHOULDER"]])
    knee_center = midpoint2D(landmarks[LANDMARK_INDEX["LEFT_KNEE"]], landmarks[LANDMARK_INDEX["RIGHT_KNEE"]])

    # make vector hip-shoulder & hip-knee
    shoulder_hip = shoulder_center - hip_center
    hip_knee     = hip_center - shoulder_center

    # make angle with hip-shoulder & hip-knee & return
    return cal_vector_angle(shoulder_hip, hip_knee)

# =========================================================
# +. knee valgus
# =========================================================
def get_knee_valgus(landmarks):
    # make left & right knee-ankle vector
    left_knee_ankle = np.array([])
    

# =========================================================
# 4. 판단용 특징값 생성
# =========================================================

def extract_pose_features(
    normalized_points,
    width,
    height,
):
    p = normalized_points

    return {
        "shoulder_tilt_deg":
            calculate_horizontal_tilt_pixel(
                p["LS"],
                p["RS"],
                width,
                height,
            ),

        "hip_tilt_deg":
            calculate_horizontal_tilt_pixel(
                p["LH"],
                p["RH"],
                width,
                height,
            ),

        "left_knee_alignment":
            calculate_knee_alignment(
                p["LH"],
                p["LK"],
                p["LA"],
                width,
                height,
            ),

        "right_knee_alignment":
            calculate_knee_alignment(
                p["RH"],
                p["RK"],
                p["RA"],
                width,
                height,
            ),
    }
# =========================================================
# 5. 다음 단계 전달용 결과 생성
# =========================================================

def create_pose_result(
    normalized_points,
    neck_center,
    features,
    timestamp_ms,
):
    
    point_result = {
        name: {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
        }
        for name, point in normalized_points.items()
    }

    # 각도 계산의 기준점도 함께 전달
    point_result["NECK_CENTER"] = {
        "x": float(neck_center[0]),
        "y": float(neck_center[1]),
        "z": float(neck_center[2]),
    }

    feature_result = {
        name: (
            round(value, 3)
            if value is not None
            else None
        )
        for name, value in features.items()
    }

    return {
        "timestamp_ms": timestamp_ms,
        "points": point_result,
        "features": feature_result,
    }


def create_llm_front_data(features):
    """LLM 담당자에게 전달할 정면 3개 측정값 딕셔너리."""
    return {
        "front": {
            "shoulder_tilt": round(features["shoulder_tilt_deg"], 1)
            if features["shoulder_tilt_deg"] is not None else None,
            "hip_tilt": round(features["hip_tilt_deg"], 1)
            if features["hip_tilt_deg"] is not None else None,
            "knee_alignment": {
                "left": round(features["left_knee_alignment"], 1)
                if features["left_knee_alignment"] is not None else None,
                "right": round(features["right_knee_alignment"], 1)
                if features["right_knee_alignment"] is not None else None,
            },
        }
    }


# =========================================================
# 6. 디버깅 화면 출력
# =========================================================

def draw_debug_pose(
    frame,
    normalized_points,
    neck_center,
    width,
    height,
):
    pixel_points = {
        name: array_to_pixel(
            point,
            width,
            height,
        )
        for name, point in normalized_points.items()
    }

    neck_pixel = array_to_pixel(
        neck_center,
        width,
        height,
    )

 
    pixel_points["NECK_CENTER"] = neck_pixel

    # 핵심 포인트 표시
    for name in normalized_points:
        pixel_point = pixel_points[name]

        cv2.circle(
            frame,
            pixel_point,
            7,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            frame,
            name,
            (
                pixel_point[0] + 8,
                pixel_point[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 가상 목 중심 표시
    cv2.circle(
        frame,
        neck_pixel,
        5,
        (255, 0, 255),
        -1,
    )

    cv2.putText(
        frame,
        "NECK",
        (
            neck_pixel[0] + 8,
            neck_pixel[1],
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # 연결선 표시
    for (
        start_name,
        end_name,
        color,
        thickness,
    ) in POSE_CONNECTIONS:
        cv2.line(
            frame,
            pixel_points[start_name],
            pixel_points[end_name],
            color,
            thickness,
        )


def draw_feature_values(frame, features):
    """
    각도 계산 결과를 디버깅 화면 왼쪽에 표시한다.
    """
    y_position = 30

    for label, key in FEATURE_DISPLAY:
        value = features.get(key)

        if value is None:
            text = f"{label}: None"
        else:
            text = f"{label}: {value:.1f}"

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

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없습니다."
        )

    start_time = time.perf_counter()
    previous_timestamp_ms = -1
    frame_count = 0

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
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000
                )

                if timestamp_ms <= previous_timestamp_ms:
                    timestamp_ms = (
                        previous_timestamp_ms + 1
                    )

                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    normalized_points, neck_center = (
                        create_pose_points(
                            landmarks
                        )
                    )

                    features = extract_pose_features(
                        normalized_points,
                        width,
                        height,
                    )

                    pose_result = create_pose_result(
                        normalized_points,
                        neck_center,
                        features,
                        timestamp_ms,
                    )

                    draw_debug_pose(
                        frame,
                        normalized_points,
                        neck_center,
                        width,
                        height,
                    )

                    draw_feature_values(
                        frame,
                        features,
                    )

                    # 30프레임마다 핵심 특징값만 출력
                    if frame_count % 30 == 0:
                        print(features)

                   

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

                frame_count += 1

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("c") and result.pose_landmarks:
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