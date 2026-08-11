import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from upper_body import (
    calculate_fha,
    classify_fha,
    calculate_fsa,
    classify_fsa,
    calculate_shoulder_tilt,
    classify_shoulder_tilt,
    calculate_thoracic_kyphosis,
    classify_thoracic_kyphosis,
)

from lower_body import (
    calculate_pelvic_tilt_ant,
    classify_pelvic_tilt_ant,
    calculate_knee_alignment,
)

from fha_ai import FHAClassifier

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 모델 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runtime_models"
    / "pose_landmarker_full.task"
)


# =========================================================
# 0. 설정값
# =========================================================

POINT_MAPPING = {
    "LEAR": 7,
    "REAR": 8,

    "LS": 11,
    "RS": 12,

    "LE": 13,
    "RE": 14,

    "LW": 15,
    "RW": 16,

    "LH": 23,
    "RH": 24,

    "LK": 25,
    "RK": 26,

    "LA": 27,
    "RA": 28,
}


LEFT_EAR_INDEX = 7
RIGHT_EAR_INDEX = 8

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12


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
    # ===== upper_body.py =====
    ("FHA", "fha_deg"),
    ("FSA", "fsa_deg"),
    ("Thoracic Kyphosis", "thoracic_kyphosis_deg"),
    ("Shoulder tilt", "shoulder_tilt_deg"),

    # ===== lower_body.py =====
    ("Pelvic tilt Ant", "pelvic_tilt_ant_deg"),
    ("Left knee alignment", "left_knee_alignment"),
    ("Right knee alignment", "right_knee_alignment"),
]


# =========================================================
# 1. 좌표 변환 및 가상점 생성
# =========================================================

def array_to_pixel(point, width, height):
    """정규화 좌표 [x, y]를 OpenCV 픽셀 좌표로 변환한다."""

    return (
        int(point[0] * width),
        int(point[1] * height),
    )


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

def calculate_horizontal_tilt_pixel(
    point_a,
    point_b,
    width,
    height,
):
    """두 점을 잇는 선의 수평축 기준 기울기를 도 단위로 계산한다."""

    ax = float(point_a[0] * width)
    ay = float(point_a[1] * height)

    bx = float(point_b[0] * width)
    by = float(point_b[1] * height)

    dx = bx - ax
    dy = -(by - ay)

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    if angle > 90.0:
        angle -= 180.0

    elif angle < -90.0:
        angle += 180.0

    return float(angle)


# =========================================================
# 3. 핵심 포인트 생성
# =========================================================

def create_pose_points(landmarks):
    """
    측정과 디버깅에 사용할 좌표를 하나의 딕셔너리로 만든다.

    HEAD:
        양쪽 귀의 중간점

    NECK_CENTER:
        양쪽 어깨의 중간점

    나머지 점:
        MediaPipe 랜드마크의 정규화된 x, y 좌표
    """

    points = {
        name: np.array(
            [
                landmarks[index].x,
                landmarks[index].y,
            ],
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
# 4. FHA Rule + AI Fusion
# =========================================================

def fuse_fha(rule_result, ai_result):

    if rule_result == "RULE_ABNORMAL":
        return "FUSION_ABNORMAL"

    if ai_result == "AI_ABNORMAL":
        return "FUSION_ABNORMAL"

    if (
        rule_result == "RULE_NORMAL"
        and ai_result == "AI_NORMAL"
    ):
        return "FUSION_NORMAL"

    return "FUSION_BORDERLINE"


# =========================================================
# 5. 다음 단계 전달용 결과 생성
# =========================================================

def create_pose_result(
    points,
    features,
    timestamp_ms,
):
    """2차원 좌표와 특징값을 직렬화 가능한 딕셔너리로 만든다."""

    point_result = {
        name: {
            "x": float(point[0]),
            "y": float(point[1]),
        }
        for name, point in points.items()
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


def create_llm_front_data(
    features,
    statuses,
    fha_ai_result,
    fha_fusion_result,
):
    """LLM 담당자에게 전달할 자세 측정값 딕셔너리."""

    return {
        "front": {

            "shoulder_tilt": {
                "angle": (
                    round(
                        features["shoulder_tilt_deg"],
                        2,
                    )
                    if features["shoulder_tilt_deg"] is not None
                    else None
                ),
                "rule": statuses["shoulder_tilt"],
            },

            "hip_tilt": (
                round(
                    features["hip_tilt_deg"],
                    2,
                )
                if features["hip_tilt_deg"] is not None
                else None
            ),

            "knee_alignment": {

                "left": (
                    round(
                        features["left_knee_alignment"],
                        2,
                    )
                    if features["left_knee_alignment"] is not None
                    else None
                ),

                "right": (
                    round(
                        features["right_knee_alignment"],
                        2,
                    )
                    if features["right_knee_alignment"] is not None
                    else None
                ),
            },
        },

        "side": {

            "fha": {

                "angle": (
                    round(
                        features["fha_deg"],
                        2,
                    )
                    if features["fha_deg"] is not None
                    else None
                ),

                "rule": statuses["fha"],

                "ai_score": round(
                    fha_ai_result["score"],
                    4,
                ),

                "ai_result":
                    fha_ai_result["result"],

                "fusion":
                    fha_fusion_result,
            },

            "fsa": {

                "angle": (
                    round(
                        features["fsa_deg"],
                        2,
                    )
                    if features["fsa_deg"] is not None
                    else None
                ),

                "rule":
                    statuses["fsa"],
            },

            "thoracic_kyphosis": {

                "angle": (
                    round(
                        features["thoracic_kyphosis_deg"],
                        2,
                    )
                    if features["thoracic_kyphosis_deg"] is not None
                    else None
                ),

                "rule":
                    statuses["thoracic_kyphosis"],
            },

            "pelvic_tilt_ant": {

                "angle": (
                    round(
                        features["pelvic_tilt_ant_deg"],
                        2,
                    )
                    if features["pelvic_tilt_ant_deg"] is not None
                    else None
                ),

                "rule":
                    statuses["pelvic_tilt_ant"],
            },
        },
    }

# =========================================================
# 6. 디버깅 화면 출력
# =========================================================

def draw_debug_pose(
    frame,
    points,
    width,
    height,
):
    """필요한 점과 연결선만 화면에 표시한다."""

    pixel_points = {
        name: array_to_pixel(
            point,
            width,
            height,
        )
        for name, point in points.items()
    }

    for start_name, end_name in POSE_CONNECTIONS:

        if (
            start_name not in pixel_points
            or end_name not in pixel_points
        ):
            continue

        cv2.line(
            frame,
            pixel_points[start_name],
            pixel_points[end_name],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for name, pixel_point in pixel_points.items():

        radius = (
            6
            if name in ("HEAD", "NECK_CENTER")
            else 4
        )

        cv2.circle(
            frame,
            pixel_point,
            radius,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_feature_values(
    frame,
    features,
):
    """측정 결과를 디버깅 화면 왼쪽에 표시한다."""

    y_position = 30

    for label, key in FEATURE_DISPLAY:

        value = features.get(key)

        text = (
            f"{label}: None"
            if value is None
            else f"{label}: {value:.1f}"
        )

        cv2.putText(
            frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        y_position += 28


# =========================================================
# 7. Main
# =========================================================

def main():

    # -----------------------------------------------------
    # MediaPipe 모델 확인
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )


    # -----------------------------------------------------
    # MediaPipe Pose Landmarker 설정
    # -----------------------------------------------------

    options = vision.PoseLandmarkerOptions(

        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=vision.RunningMode.VIDEO,

        num_poses=1,

        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,

        output_segmentation_masks=False,
    )


    # -----------------------------------------------------
    # FHA AI TensorRT
    # -----------------------------------------------------

    fha_ai = FHAClassifier()


    # -----------------------------------------------------
    # 카메라
    # -----------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "카메라를 열 수 없습니다."
        )


    # -----------------------------------------------------
    # MediaPipe VIDEO timestamp
    # -----------------------------------------------------

    start_time = time.perf_counter()

    previous_timestamp_ms = -1


    try:

        with vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                success, frame = cap.read()

                if not success:

                    print(
                        "프레임을 읽지 못했습니다."
                    )

                    break


                height, width = frame.shape[:2]


                # -------------------------------------------------
                # OpenCV BGR → MediaPipe RGB
                # -------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )


                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )


                # -------------------------------------------------
                # timestamp 생성
                # -------------------------------------------------

                timestamp_ms = int(
                    (
                        time.perf_counter()
                        - start_time
                    )
                    * 1000
                )


                if timestamp_ms <= previous_timestamp_ms:

                    timestamp_ms = (
                        previous_timestamp_ms + 1
                    )


                previous_timestamp_ms = timestamp_ms


                # -------------------------------------------------
                # MediaPipe Pose 추론
                # -------------------------------------------------

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )


                features = None
                fha_ai_result = None
                fha_fusion_result = None


                # -------------------------------------------------
                # Pose 검출 성공
                # -------------------------------------------------

                if result.pose_landmarks:

                    landmarks = (
                        result.pose_landmarks[0]
                    )


                    points = create_pose_points(
                        landmarks
                    )


                    # =============================================
                    # Knee alignment
                    # =============================================

                    left_knee_result = (
                        calculate_knee_alignment(
                            points["LH"],
                            points["LK"],
                            points["LA"],
                            width,
                            height,
                            "left",
                        )
                    )


                    right_knee_result = (
                        calculate_knee_alignment(
                            points["RH"],
                            points["RK"],
                            points["RA"],
                            width,
                            height,
                            "right",
                        )
                    )


                    # =============================================
                    # 자세 특징값 계산
                    # =============================================

                    features = {

                        # -----------------------------------------
                        # 정면
                        # -----------------------------------------

                        "hip_tilt_deg":
                            calculate_horizontal_tilt_pixel(
                                points["LH"],
                                points["RH"],
                                width,
                                height,
                            ),


                        "left_knee_alignment": (
                            left_knee_result["angle"]
                            if left_knee_result is not None
                            else None
                        ),


                        "right_knee_alignment": (
                            right_knee_result["angle"]
                            if right_knee_result is not None
                            else None
                        ),


                        # -----------------------------------------
                        # upper_body.py
                        # -----------------------------------------

                        "fha_deg": calculate_fha(
                            points["REAR"],
                            points["RS"],
                           
                        ),


                        "fsa_deg": calculate_fsa(
                            points["NECK_CENTER"],
                            points["LS"],
                            
                        ),


                        "shoulder_tilt_deg":
                            calculate_shoulder_tilt(
                                points["LS"],
                                points["RS"],
                                
                            ),


                        "thoracic_kyphosis_deg":
                            calculate_thoracic_kyphosis(
                                points["HEAD"],
                                points["LS"],
                                points["LH"],
                                
                            ),


                        # -----------------------------------------
                        # lower_body.py
                        # -----------------------------------------

                        "pelvic_tilt_ant_deg":
                            calculate_pelvic_tilt_ant(
                                points["LS"],
                                points["RS"],
                                points["LH"],
                                points["RH"],
                                points["LK"],
                                points["RK"],
                                width,
                                height,
                            ),
                    }


                    # =============================================
                    # FHA AI 추론
                    # =============================================

                    fha_ai_result = fha_ai.predict_result(
                        frame
                    )


                    # =============================================
                    # Rule 기반 분류
                    # =============================================

                    statuses = {

                        "fha":
                            classify_fha(
                                features["fha_deg"]
                            ),


                        "fsa":
                            classify_fsa(
                                features["fsa_deg"]
                            ),


                        "shoulder_tilt":
                            classify_shoulder_tilt(
                                features[
                                    "shoulder_tilt_deg"
                                ]
                            ),


                        "thoracic_kyphosis":
                            classify_thoracic_kyphosis(
                                features[
                                    "thoracic_kyphosis_deg"
                                ]
                            ),


                        "pelvic_tilt_ant":
                            classify_pelvic_tilt_ant(
                                features[
                                    "pelvic_tilt_ant_deg"
                                ]
                            ),


                        "left_knee": (
                            left_knee_result["direction"]
                            if left_knee_result is not None
                            else "measurement_failed"
                        ),


                        "right_knee": (
                            right_knee_result["direction"]
                            if right_knee_result is not None
                            else "measurement_failed"
                        ),
                    }


                    # =============================================
                    # FHA Rule + AI Fusion
                    # =============================================

                    fha_fusion_result = fuse_fha(
                    statuses["fha"],
                    fha_ai_result["result"],
                )

                    fha_result = {
                        "angle": features["fha_deg"],
                        "rule": statuses["fha"],
                        "ai_score": fha_ai_result["score"],
                        "ai": fha_ai_result["result"],
                        "fusion": fha_fusion_result,
                    }


                    # =============================================
                    # 디버깅 표시
                    # =============================================

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


                # -------------------------------------------------
                # Pose 미검출
                # -------------------------------------------------

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


                # -------------------------------------------------
                # 화면 출력
                # -------------------------------------------------

                cv2.imshow(
                    "Static Front Posture Measurement",
                    frame,
                )


                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )


                if key == ord("q"):

                    break


                # -------------------------------------------------
                # 결과 캡처
                # -------------------------------------------------

                if (
                    key == ord("c")
                    and features is not None
                ):

                    print(
                        f'FHA Rule: '
                        f'{features["fha_deg"]:.2f} / '
                        f'{statuses["fha"]}'
                    )

                    print(
                        f'FHA AI: '
                        f'{fha_ai_result["score"]:.4f} / '
                        f'{fha_ai_result["result"]}'
                    )

                    print(
                        f'FHA Fusion: '
                        f'{fha_fusion_result}'
                    )


                    llm_data = (
                        create_llm_front_data(
                            features,
                            statuses,
                            fha_ai_result,
                            fha_fusion_result,
                        )
                    )


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