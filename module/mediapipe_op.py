import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from basic_fn import *



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

        elif key == 255: # no key input check
            self.key_released = True

        return None