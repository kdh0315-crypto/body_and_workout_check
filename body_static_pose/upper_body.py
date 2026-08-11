import math
import numpy as np


# ========================================================
# +. FHA
# ========================================================

def calculate_fha(ear, shoulder):
    ear_x = float(ear[0])
    ear_y = float(ear[1])

    shoulder_x = float(shoulder[0])
    shoulder_y = float(shoulder[1])

    dx = abs(ear_x - shoulder_x)
    dy = abs(ear_y - shoulder_y)

    if dx < 1e-6 and dy < 1e-6:
        return None

    if dy <= 0:
        return None

    return float(math.degrees(math.atan2(dx, dy)))


def classify_fha(fha_deg):
    if fha_deg is None:
        return "measurement_failed"

    if fha_deg < 32.0:
        return "RULE_NORMAL"

    if fha_deg < 36.0:
        return "RULE_BORDERLINE"

    return "RULE_ABNORMAL"


# ========================================================
# +. FSA
# ========================================================

def calculate_fsa(neck_center, shoulder):
    neck_x = float(neck_center[0])
    neck_y = float(neck_center[1])

    shoulder_x = float(shoulder[0])
    shoulder_y = float(shoulder[1])

    dx = abs(shoulder_x - neck_x)
    dy = shoulder_y - neck_y

    if dx < 1e-6 and abs(dy) < 1e-6:
        return None

    if dy <= 0:
        return None

    return float(math.degrees(math.atan2(dx, dy)))


def classify_fsa(fsa_deg):
    if fsa_deg is None:
        return "measurement_failed"

    if fsa_deg >= 52.0:
        return "abnormal"

    return "normal"


# ========================================================
# +. Shoulder tilt angle
# ========================================================

def calculate_shoulder_tilt(left_shoulder, right_shoulder):
    left_x = float(left_shoulder[0])
    left_y = float(left_shoulder[1])

    right_x = float(right_shoulder[0])
    right_y = float(right_shoulder[1])

    dx = abs(right_x - left_x)
    dy = right_y - left_y

    if dx < 1e-6 and abs(dy) < 1e-6:
        return None

    return float(abs(math.degrees(math.atan2(dy, dx))))


def classify_shoulder_tilt(angle):
    if angle is None:
        return "measurement_failed"

    if abs(angle) > 2.5:
        return "abnormal"

    return "normal"


# ========================================================
# +. Thoracic kyphosis
# ========================================================

def calculate_thoracic_kyphosis(
    head,
    shoulder,
    hip,
):
    shoulder_xy = np.array(shoulder[:2], dtype=np.float32)
    hip_xy = np.array(hip[:2], dtype=np.float32)

    # shoulder to hip midpoint
    hs_midpoint = (shoulder_xy + hip_xy) / 2.0

    head_x = float(head[0])
    head_y = float(head[1])

    hs_x = float(hs_midpoint[0])
    hs_y = float(hs_midpoint[1])

    dx = abs(head_x - hs_x)
    dy = hs_y - head_y

    if dx < 1e-6 and abs(dy) < 1e-6:
        return None

    if dy <= 0:
        return None

    return float(math.degrees(math.atan2(dy, dx)))


def classify_thoracic_kyphosis(angle):
    if angle is None:
        return "measurement_failed"

    if angle > 40.0:
        return "abnormal"

    if 20.0 <= angle <= 40.0:
        return "normal"

    return "out_of_range"