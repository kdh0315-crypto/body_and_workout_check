import mediapipe as mp
import cv2
import numpy as np

from module.basic_fn import *
from upper_body import *
from lower_body import *


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
    ####from upper.py####
    ("FHA", "fha_deg"),
    ("FSA", "fsa_deg"),
    ("Thoracic Kyphosis", "thoracic_kyphosis_deg"),
    ("Shoulder tilt", "shoulder_tilt_deg"),
    #### from lower_body.py ####
    ("Pelvic tilt Ant", "pelvic_tilt_ant_deg"),
    ("Left knee valgus", "left_knee_valgus_deg"),
    ("Right knee valgus", "right_knee_valgus_deg"),
]

def get_landmark_xy(landmarks, landmark_enum, image_width, image_height):
    """정규화된 좌표(0~1)를 실제 픽셀 좌표로 변환해서 반환"""
    lm = landmarks[landmark_enum.value]
    return [lm.x * image_width, lm.y * image_height], lm.visibility


class CaptureForm():
    def __init__(self):
        # set initial image to None
        self.front_img = None
        self.side_img = None

        # enable signal to check each image is captured
        self.front_img_en = 0
        self.side_img_en  = 0

        # dont signal to make other module can check operation done easily
        self.done = False

        # check key is released to do not check front & side data simultaniously
        self.key_released = True

    def capture_form(self, key, frame):
        """
        Rule. Capture front image first
        """
        if key == ord('c') and self.key_released == True:
            if self.front_img_en == 0:
                self.front_img = frame.copy()
                self.key_released = False
                self.front_img_en = 1
                print("Front Image Capture complete")

            elif self.side_img_en == 0:
                self.side_img = frame.copy()
                self.key_released = False
                self.side_img_en = 1
                self.done = True
                print("Side Image Capture complete")
                return self.front_img, self.side_img

        elif key != 255: # no key input check
            self.key_released = True

        return None

    def get_joint_point(self):
        None

    def get_body_angle(self):
        # Front image
        # upper body
        shoulder_tilt = calculate_shoulder_tilt(self.front_img)
        # lower body
        knee_valgus = calculate_knee_valgus_angle()