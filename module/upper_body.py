import math


# ========================================================
# +. FHA
# ========================================================
def calculate_fha(ear, shoulder, hip, width=None, height=None):
    """EAR-SHOULDER vector angle relative to HIP-SHOULDER trunk vector."""

    ear_x = float(ear[0])
    ear_y = float(ear[1])

    shoulder_x = float(shoulder[0])
    shoulder_y = float(shoulder[1])

    hip_x = float(hip[0])
    hip_y = float(hip[1])

    # normalized coordinates -> pixel coordinates
    if width is not None and height is not None:
        ear_x *= width
        ear_y *= height

        shoulder_x *= width
        shoulder_y *= height

        hip_x *= width
        hip_y *= height

    # Hip -> Shoulder : 몸통 기준 벡터
    trunk_x = shoulder_x - hip_x
    trunk_y = shoulder_y - hip_y

    # Shoulder -> Ear : 머리 방향 벡터
    head_x = ear_x - shoulder_x
    head_y = ear_y - shoulder_y

    trunk_norm = math.hypot(trunk_x, trunk_y)
    head_norm = math.hypot(head_x, head_y)

    if trunk_norm < 1e-6 or head_norm < 1e-6:
        return None

    # 두 벡터의 내적
    dot = trunk_x * head_x + trunk_y * head_y

    cos_theta = dot / (trunk_norm * head_norm)

    # 부동소수점 오차 방지
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return float(math.degrees(math.acos(cos_theta)))


def classify_fha(fha_deg):
    if fha_deg is None:
        return "measurement_failed"

    if fha_deg < 32.0:
        return "RULE_NORMAL"

    if fha_deg < 36.0:
        return "RULE_BORDERLINE"

    return "RULE_ABNORMAL"


     
#shoulder tilt angle
def calculate_shoulder_tilt(left_shoulder, right_shoulder, width, height):
    left_x = left_shoulder[0] * width
    left_y = left_shoulder[1] * height

    right_x = right_shoulder[0] * width
    right_y = right_shoulder[1] * height

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
# +.thoracic kyphosis
# ========================================================

def calculate_thoracic_kyphosis(
    head,
    shoulder,
    hip,
    width,
    height,
):
    hs_midpoint = (shoulder + hip) / 2.0

    head_x = float(head[0] * width)
    head_y = float(head[1] * height)

    hs_x = float(hs_midpoint[0] * width)
    hs_y = float(hs_midpoint[1] * height)

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

    if angle >= 80.0:
        return "normal"

    if angle >= 70.0:
        return "borderline"

    return "abnormal"