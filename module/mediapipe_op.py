import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from module.basic_fn import *


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
MODEL_PATH = "module/models/pose_landmarker_full.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# ===== Landmarker for Image =====
image_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
)
image_landmarker = vision.PoseLandmarker.create_from_options(image_options)

# ===== Landmarker for Video =====
video_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
video_landmarker = vision.PoseLandmarker.create_from_options(video_options)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ===== Landmark point name =====
KEY_LANDMARKS = {
    "nose":           0,
    "left_ear":       7,
    "right_ear":      8,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}


# =====================================
# Extract pose landmark
# =====================================
# ===== From Image =====
def extract_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = image_landmarker.detect(mp_image)
    return result

# ===== From Video =====
def extract_landmarks_video(image, timestamp_ms):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return video_landmarker.detect_for_video(mp_image, timestamp_ms)


# Check existance of landmark
def has_landmarks(results):
    return results.pose_landmarks is not None and len(results.pose_landmarks) > 0

# =====================================
# Convert Tasks result -> drawing proto (스켈레톤 그리기용)
# =====================================
def _to_landmark_proto(results):
    """Tasks 랜드마크를 draw_landmarks가 받는 proto 형식으로 변환."""
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in results.pose_landmarks[0]
    ])
    return proto

# =====================================
# Draw landmarks on Image
# =====================================
def draw_skeleton(image, results):
    annotated = image.copy()
    if has_landmarks(results):
        mp_drawing.draw_landmarks(
            annotated,
            _to_landmark_proto(results),
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
    if has_landmarks(results):
        mp_drawing.draw_landmarks(
            white,
            _to_landmark_proto(results),
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
    if not has_landmarks(results):
        print(f"{filename}: Failed to detect pose")
    annotated = draw_skeleton_on_white(image, results) if on_white else draw_skeleton(image, results)
    cv2.imwrite(filename, annotated)
    return results


# =====================================
# Get landmark pixel coordinate into dict type
# =====================================
def get_landmark_pixels(results, image_shape):
    """{landmark num: (x, y, visibility)}"""
    if not has_landmarks(results):
        return None

    h, w = image_shape[:2]
    coords = {}
    for idx, lm in enumerate(results.pose_landmarks[0]):
        coords[idx] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return coords


# =====================================
# Get landmark pixel coordinate based on pose name
# =====================================
def get_pose_point(results, image_shape):
    coords = get_landmark_pixels(results, image_shape)
    if coords is None:
        return None

    points = {}
    for name, idx in KEY_LANDMARKS.items():
        points[name] = coords[idx]
    return points

# =====================================
# Get landmark normalized coordinate based on pose name
# =====================================
def get_pose_point_norm(results, with_vis=False):
    """정규화 좌표(0~1) 반환. with_vis=True면 [x, y, visibility]."""
    if not has_landmarks(results):
        return None

    lm_list = results.pose_landmarks[0]
    points = {}
    for name, idx in KEY_LANDMARKS.items():
        lm = lm_list[idx]
        if with_vis:
            points[name] = [lm.x, lm.y, getattr(lm, "visibility", 1.0)]
        else:
            points[name] = [lm.x, lm.y]
    return points