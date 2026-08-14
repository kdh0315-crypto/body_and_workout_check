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
    right_shoulder,
    right_hip,
    right_knee,
    width,
    height,
):
    # normalized coordinate -> pixel coordinate
    right_shoulder_px = np.array([
        right_shoulder[0] * width,
        right_shoulder[1] * height,
    ], dtype=np.float32)

    right_hip_px = np.array([
        right_hip[0] * width,
        right_hip[1] * height,
    ], dtype=np.float32)

    right_knee_px = np.array([
        right_knee[0] * width,
        right_knee[1] * height,
    ], dtype=np.float32)

    # hip as vertex
    hip_to_shoulder = right_shoulder_px - right_hip_px
    hip_to_knee = right_knee_px - right_hip_px

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

    if angle <= 18.0:
        return "normal"

    return "abnormal"


# ========================================================
# +. Knee valgus angle
# ========================================================

def calculate_knee_valgus_angle(
    hip,
    knee,
    ankle,
    width,
    height,
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

    # knee -> hip vector
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
    return float(180.0 - internal_angle)