import mediapipe as mp
import cv2
import numpy as np

from module.basic_fn import *
# from upper_body import *
# from lower_body import *


# =====================================
# Capture Image to check body form
# =====================================
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

# =====================================
# Initialize Mediapipe
# =====================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# Get landmark coordinate
# =====================================
def get_landmark_xy(landmarks, landmark_enum, image_width, image_height):
    """Return normalized coordinate into real pixel coordinate"""
    lm = landmarks[landmark_enum.value]
    return [lm.x * image_width, lm.y * image_height], lm.visibility


# =====================================
# Extract pose landmark from image
# =====================================
def extract_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = pose.process(rgb)
    return result

# =====================================
# Draw landmarks on Image
# =====================================
def draw_skeleton(image, results):
    annotated = image.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
    return annotated


# =====================================
# Draw landmarks on white image - just want to see skeleton
# =====================================
def draw_skeleton_on_white(image, results):
    white = np.ones_like(image) * 255
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            white,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
    return white


# =====================================
# Save skeleton only image
# =====================================
def save_with_pose(image, filename, on_white=False):
    results = extract_landmarks(image)
    if results.pose_landmarks is None:
        print(f"{filename}: Failed to detect pose")
    annotated = draw_skeleton_on_white(image, results) if on_white else draw_skeleton(image, results)
    cv2.imwrite(filename, annotated)
    return results


# =====================================
# Get landmark pixel coordinate into dict type
# =====================================
def get_landmark_pixels(results, image_shape):
    """{num: (x, y, visibility)}"""
    if results.pose_landmarks is None:
        return None
    h, w = image_shape[:2]
    coords = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        coords[idx] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return coords