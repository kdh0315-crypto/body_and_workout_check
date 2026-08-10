"""
운동 자세 인식 - 16단계 (TensorRT 전환)
스쿼트, 목 스트레칭, 바이셉 컬, 플랭크 - 학습 모델을 tensorflow(.h5) 대신
TensorRT 엔진(.trt)으로 불러와 추론하도록 전면 교체.

변경사항 (15단계 -> 16단계):
- import tensorflow 제거 (mediapipe와의 버전 충돌 문제를 원천 차단)
- import tensorrt, pycuda 추가
- TRTModel 클래스 신규 추가: .trt 엔진을 불러와 tensorflow의 model.predict()처럼
  사용할 수 있게 감싸는 공통 클래스
- BicepCurlChecker, PlankChecker: 모델 로드를 tf.keras.models.load_model() 대신
  TRTModel(...)로 교체, predict() 호출부 입력 형태만 맞춰서 수정
- SquatChecker: 카메라 방향 감지(front/side) + 하이브리드 판별 구조로 확장
  (이번 단계에서 함께 반영)

실행 전 설치 (이미 완료됨, 재확인용):
    pip install mediapipe opencv-python numpy
    (tensorflow 설치 불필요 - tensorrt, pycuda는 이미 Jetson에 설치되어 있음)

실행:
    python squat_pose_test.py

조작:
    'r' 키 - 현재 운동 세션 리셋
    'q' 키 - 종료
"""

import math
import time
import cv2
import mediapipe as mp
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (GPU 컨텍스트 자동 초기화, import만으로 의미 있음)

# -----------------------------
# MediaPipe 초기화
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# TensorRT 추론 공통 클래스
# .trt 엔진 파일을 불러와서, tensorflow의 model.predict()와 비슷하게
# 쓸 수 있도록 감싸주는 클래스. 모든 운동 Checker가 공용으로 재사용.
# -----------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTModel:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        input_shape = self.engine.get_tensor_shape(self.input_name)
        output_shape = self.engine.get_tensor_shape(self.output_name)

        self.input_size = int(np.prod(input_shape))
        self.output_size = int(np.prod(output_shape))

        self.d_input = cuda.mem_alloc(self.input_size * np.float32().itemsize)
        self.d_output = cuda.mem_alloc(self.output_size * np.float32().itemsize)

        self.stream = cuda.Stream()

    def predict(self, input_array):
        """
        input_array: 1차원 또는 2차원 numpy array (각도값 몇 개)
        반환: numpy array (모델 출력 확률값들)
        """
        input_array = np.ascontiguousarray(input_array, dtype=np.float32)
        output_array = np.empty(self.output_size, dtype=np.float32)

        cuda.memcpy_htod_async(self.d_input, input_array, self.stream)

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(output_array, self.d_output, self.stream)
        self.stream.synchronize()

        return output_array


# -----------------------------
# 학습된 TensorRT 엔진 로드 (실제 파일 경로에 맞게 수정해서 사용)
# -----------------------------
BICEP_CURL_MODEL_PATH = "bicep_curl_classifier.trt"
bicep_curl_model = TRTModel(BICEP_CURL_MODEL_PATH)

PLANK_MODEL_PATH = "plank_classifier.trt"
plank_model = TRTModel(PLANK_MODEL_PATH)

SQUAT_MODEL_PATH = "squat_classifier.trt"
squat_model = TRTModel(SQUAT_MODEL_PATH)


# -----------------------------
# 공통 각도 계산 함수 (모든 운동이 재사용 가능)
# -----------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_trunk_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return angle


def calculate_horizontal_tilt(point_a, point_b):
    """
    두 점을 잇는 선이 수평축에서 얼마나 기울었는지 계산 (도 단위).
    화면 좌표는 아래로 갈수록 y가 증가하므로 부호를 반전해서 계산.
    결과는 -90~90도로 정규화됨. 0도에 가까우면 수평.
    """
    dx = point_b[0] - point_a[0]
    dy = -(point_b[1] - point_a[1])

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    angle = math.degrees(math.atan2(dy, dx))

    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0

    return angle


def calculate_neck_tilt(left_eye, right_eye, left_shoulder, right_shoulder):
    eye_angle = calculate_horizontal_tilt(left_eye, right_eye)
    shoulder_angle = calculate_horizontal_tilt(left_shoulder, right_shoulder)
    return eye_angle - shoulder_angle


def calculate_hip_deviation(shoulder, hip, ankle):
    """
    어깨-발목을 잇는 직선(가상의 몸통 라인) 대비, 엉덩이가 위/아래로
    얼마나 벗어났는지 계산.
    양수: 엉덩이가 라인보다 아래로 처짐 (Low back)
    음수: 엉덩이가 라인보다 위로 솟음 (High back)
    """
    if ankle[0] == shoulder[0]:
        expected_hip_y = (shoulder[1] + ankle[1]) / 2
    else:
        t = (hip[0] - shoulder[0]) / (ankle[0] - shoulder[0])
        expected_hip_y = shoulder[1] + t * (ankle[1] - shoulder[1])
    return hip[1] - expected_hip_y


def get_landmark_xy(landmarks, landmark_enum, image_width, image_height):
    """정규화된 좌표(0~1)를 실제 픽셀 좌표로 변환해서 반환"""
    lm = landmarks[landmark_enum.value]
    return [lm.x * image_width, lm.y * image_height], lm.visibility


# -----------------------------
# 운동 세션 관리 클래스 (모든 운동이 공통으로 재사용)
# -----------------------------
class ExerciseSession:
    def __init__(self, target_reps):
        self.target_reps = target_reps
        self.reset()

    def reset(self):
        self.rep_count = 0
        self.last_rep_errors = []
        self.has_completed_rep = False
        self.error_counter = {}
        self.set_completed = False
        print("[RESET] Session restarted.")

    def record_rep(self, errors):
        self.rep_count += 1
        self.last_rep_errors = errors
        self.has_completed_rep = True

        if errors:
            for err in errors:
                self.error_counter[err] = self.error_counter.get(err, 0) + 1
        else:
            self.error_counter["Good form"] = self.error_counter.get("Good form", 0) + 1

        print(f"[REP COUNT] {self.rep_count}/{self.target_reps}  errors={errors if errors else 'None'}")

        if self.rep_count >= self.target_reps:
            self.set_completed = True
            print("=" * 40)
            print(f"[SET COMPLETE] {self.target_reps} reps finished. Summary:")
            for err_name, count in self.error_counter.items():
                print(f"  - {err_name}: {count} time(s)")
            print("=" * 40)


# -----------------------------
# 스쿼트 판별 로직 (SquatChecker)
# 측면 전용 규칙 기반 판별 (기존 로직 그대로 유지)
# -----------------------------
class SquatChecker:
    name = "squat"

    STATE_STANDING = "STANDING"
    STATE_DESCENDING = "DESCENDING"
    STATE_BOTTOM = "BOTTOM"
    STATE_ASCENDING = "ASCENDING"

    STANDING_THRESHOLD = 160
    BOTTOM_THRESHOLD = 110

    DEPTH_ANGLE_THRESHOLD = 100
    TRUNK_ANGLE_THRESHOLD = 40
    HIP_FLEX_THRESHOLD = 90

    def __init__(self, target_reps=10):
        self.session = ExerciseSession(target_reps)
        self.current_state = self.STATE_STANDING
        self.prev_knee_angle = 180
        self.bottom_snapshot_errors = []

    def reset(self):
        self.current_state = self.STATE_STANDING
        self.prev_knee_angle = 180
        self.bottom_snapshot_errors = []
        self.session.reset()

    def _update_state(self, knee_angle):
        current_state = self.current_state
        prev_knee_angle = self.prev_knee_angle

        if knee_angle > self.STANDING_THRESHOLD:
            return self.STATE_STANDING

        if current_state == self.STATE_STANDING:
            if knee_angle < prev_knee_angle:
                return self.STATE_DESCENDING
            return self.STATE_STANDING

        if current_state == self.STATE_DESCENDING:
            if knee_angle <= self.BOTTOM_THRESHOLD:
                return self.STATE_BOTTOM
            return self.STATE_DESCENDING

        if current_state == self.STATE_BOTTOM:
            if knee_angle > prev_knee_angle:
                return self.STATE_ASCENDING
            return self.STATE_BOTTOM

        if current_state == self.STATE_ASCENDING:
            if knee_angle > self.STANDING_THRESHOLD:
                return self.STATE_STANDING
            return self.STATE_ASCENDING

        return current_state

    def _check_bottom_form(self, trunk_angle, knee_angle, hip_flex_angle):
        errors = []

        if trunk_angle > self.TRUNK_ANGLE_THRESHOLD:
            errors.append("Straighten your back")

        if knee_angle > self.DEPTH_ANGLE_THRESHOLD:
            errors.append("Squat deeper")

        if hip_flex_angle > self.HIP_FLEX_THRESHOLD:
            errors.append("Bend at your hips more")

        return errors

    def _select_best_side(self, landmarks, w, h):
        l_shoulder, l_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_hip, l_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_knee, l_knee_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        l_ankle, l_ankle_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)
        left_visibility_sum = l_shoulder_vis + l_hip_vis + l_knee_vis + l_ankle_vis

        r_shoulder, r_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_hip, r_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_knee, r_knee_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        r_ankle, r_ankle_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)
        right_visibility_sum = r_shoulder_vis + r_hip_vis + r_knee_vis + r_ankle_vis

        if left_visibility_sum >= right_visibility_sum:
            shoulder, hip, knee, ankle = l_shoulder, l_hip, l_knee, l_ankle
            min_vis = min(l_shoulder_vis, l_hip_vis, l_knee_vis, l_ankle_vis)
            side_used = "left"
        else:
            shoulder, hip, knee, ankle = r_shoulder, r_hip, r_knee, r_ankle
            min_vis = min(r_shoulder_vis, r_hip_vis, r_knee_vis, r_ankle_vis)
            side_used = "right"

        return shoulder, hip, knee, ankle, min_vis, side_used

    def update(self, landmarks, w, h):
        shoulder, hip, knee, ankle, min_vis, side_used = self._select_best_side(landmarks, w, h)

        min_visibility = 0.5
        if min_vis <= min_visibility:
            return None

        knee_angle = calculate_angle(hip, knee, ankle)
        trunk_angle = calculate_trunk_angle(shoulder, hip)
        hip_flex_angle = calculate_angle(shoulder, hip, knee)

        if not self.session.set_completed:
            new_state = self._update_state(knee_angle)

            if new_state != self.current_state:
                print(f"[STATE CHANGE] {self.current_state} -> {new_state}  (knee_angle={int(knee_angle)}, side={side_used})")

            rep_just_completed = (self.current_state == self.STATE_ASCENDING and new_state == self.STATE_STANDING)

            self.current_state = new_state
            self.prev_knee_angle = knee_angle

            if self.current_state == self.STATE_BOTTOM:
                self.bottom_snapshot_errors = self._check_bottom_form(trunk_angle, knee_angle, hip_flex_angle)

            if rep_just_completed:
                self.session.record_rep(self.bottom_snapshot_errors)
                self.bottom_snapshot_errors = []

        return {
            "points": {"knee": knee, "hip": hip},
            "angles": {"knee": knee_angle, "hip": hip_flex_angle, "trunk": trunk_angle},
        }


# -----------------------------
# 목 스트레칭 판별 로직 (NeckStretchChecker) - 변경 없음
# -----------------------------
class NeckStretchChecker:
    name = "neck_stretch"

    STATE_CENTER = "CENTER"
    STATE_TILT_LEFT = "TILT_LEFT"
    STATE_TILT_RIGHT = "TILT_RIGHT"

    CENTER_THRESHOLD = 10
    TILT_ENTER_THRESHOLD = 15
    TARGET_TILT_THRESHOLD = 25

    def __init__(self, target_reps=10):
        self.session = ExerciseSession(target_reps)
        self.current_state = self.STATE_CENTER
        self.peak_tilt_this_phase = 0.0
        self.done_left = False
        self.done_right = False
        self.pending_errors = []

    def reset(self):
        self.current_state = self.STATE_CENTER
        self.peak_tilt_this_phase = 0.0
        self.done_left = False
        self.done_right = False
        self.pending_errors = []
        self.session.reset()

    def _finish_side(self, side_name, peak_tilt):
        if abs(peak_tilt) < self.TARGET_TILT_THRESHOLD:
            self.pending_errors.append(f"Tilt further to the {side_name}")

    def update(self, landmarks, w, h):
        left_eye, left_eye_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_EYE, w, h)
        right_eye, right_eye_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_EYE, w, h)
        shoulder, shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        r_shoulder, r_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)

        min_visibility = 0.5
        if min(left_eye_vis, right_eye_vis, shoulder_vis, r_shoulder_vis) <= min_visibility:
            return None

        neck_tilt = calculate_neck_tilt(left_eye, right_eye, shoulder, r_shoulder)

        if not self.session.set_completed:
            if self.current_state == self.STATE_CENTER:
                if neck_tilt > self.TILT_ENTER_THRESHOLD:
                    self.current_state = self.STATE_TILT_LEFT
                    self.peak_tilt_this_phase = neck_tilt
                elif neck_tilt < -self.TILT_ENTER_THRESHOLD:
                    self.current_state = self.STATE_TILT_RIGHT
                    self.peak_tilt_this_phase = neck_tilt

            elif self.current_state == self.STATE_TILT_LEFT:
                self.peak_tilt_this_phase = max(self.peak_tilt_this_phase, neck_tilt)
                if abs(neck_tilt) < self.CENTER_THRESHOLD:
                    self._finish_side("left", self.peak_tilt_this_phase)
                    self.done_left = True
                    self.current_state = self.STATE_CENTER
                    self.peak_tilt_this_phase = 0.0

            elif self.current_state == self.STATE_TILT_RIGHT:
                self.peak_tilt_this_phase = min(self.peak_tilt_this_phase, neck_tilt)
                if abs(neck_tilt) < self.CENTER_THRESHOLD:
                    self._finish_side("right", self.peak_tilt_this_phase)
                    self.done_right = True
                    self.current_state = self.STATE_CENTER
                    self.peak_tilt_this_phase = 0.0

            if self.current_state == self.STATE_CENTER and self.done_left and self.done_right:
                self.session.record_rep(self.pending_errors)
                self.pending_errors = []
                self.done_left = False
                self.done_right = False

        return {
            "points": {"nose_ref": shoulder},
            "angles": {"neck_tilt": neck_tilt},
        }


# -----------------------------
# 바이셉 컬 판별 로직 (BicepCurlChecker)
# 모델 호출 방식만 TRTModel로 교체, 나머지 로직은 동일
# -----------------------------
class BicepCurlChecker:
    name = "bicep_curl"

    STATE_EXTENDED = "EXTENDED"
    STATE_CURLING = "CURLING"
    STATE_FLEXED = "FLEXED"
    STATE_EXTENDING = "EXTENDING"

    EXTENDED_THRESHOLD = 150
    FLEXED_THRESHOLD = 65

    TRUNK_CLEAR_OK = 8
    MODEL_CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, target_reps=10):
        self.session = ExerciseSession(target_reps)
        self.current_state = self.STATE_EXTENDED
        self.prev_elbow_angle = 170
        self.flexed_snapshot_errors = []

    def reset(self):
        self.current_state = self.STATE_EXTENDED
        self.prev_elbow_angle = 170
        self.flexed_snapshot_errors = []
        self.session.reset()

    def _update_state(self, elbow_angle):
        current_state = self.current_state
        prev_elbow_angle = self.prev_elbow_angle

        if current_state == self.STATE_EXTENDED:
            if elbow_angle < prev_elbow_angle and elbow_angle < self.EXTENDED_THRESHOLD:
                return self.STATE_CURLING
            return self.STATE_EXTENDED

        if current_state == self.STATE_CURLING:
            if elbow_angle <= self.FLEXED_THRESHOLD:
                return self.STATE_FLEXED
            return self.STATE_CURLING

        if current_state == self.STATE_FLEXED:
            if elbow_angle > prev_elbow_angle:
                return self.STATE_EXTENDING
            return self.STATE_FLEXED

        if current_state == self.STATE_EXTENDING:
            if elbow_angle >= self.EXTENDED_THRESHOLD:
                return self.STATE_EXTENDED
            return self.STATE_EXTENDING

        return current_state

    def _check_flexed_form(self, elbow_angle, trunk_angle):
        errors = []

        if trunk_angle <= self.TRUNK_CLEAR_OK:
            return errors

        # ---- TensorRT 추론 (기존 tensorflow model.predict 대체) ----
        input_array = np.array([elbow_angle, trunk_angle], dtype=np.float32)
        output = bicep_curl_model.predict(input_array)
        prob = float(output[0])

        if prob > self.MODEL_CONFIDENCE_THRESHOLD:
            errors.append("Straighten your back (lean back detected)")

        return errors

    def _select_best_side(self, landmarks, w, h):
        l_shoulder, l_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_elbow, l_elbow_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, w, h)
        l_wrist, l_wrist_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
        l_hip, l_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        left_visibility_sum = l_shoulder_vis + l_elbow_vis + l_wrist_vis + l_hip_vis

        r_shoulder, r_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_elbow, r_elbow_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        r_wrist, r_wrist_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        r_hip, r_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        right_visibility_sum = r_shoulder_vis + r_elbow_vis + r_wrist_vis + r_hip_vis

        if left_visibility_sum >= right_visibility_sum:
            shoulder, elbow, wrist, hip = l_shoulder, l_elbow, l_wrist, l_hip
            min_vis = min(l_shoulder_vis, l_elbow_vis, l_wrist_vis, l_hip_vis)
            side_used = "left"
        else:
            shoulder, elbow, wrist, hip = r_shoulder, r_elbow, r_wrist, r_hip
            min_vis = min(r_shoulder_vis, r_elbow_vis, r_wrist_vis, r_hip_vis)
            side_used = "right"

        return shoulder, elbow, wrist, hip, min_vis, side_used

    def update(self, landmarks, w, h):
        shoulder, elbow, wrist, hip, min_vis, side_used = self._select_best_side(landmarks, w, h)

        min_visibility = 0.5
        if min_vis <= min_visibility:
            return None

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        trunk_angle = calculate_trunk_angle(shoulder, hip)

        if not self.session.set_completed:
            new_state = self._update_state(elbow_angle)

            if new_state != self.current_state:
                print(f"[STATE CHANGE] {self.current_state} -> {new_state}  (elbow_angle={int(elbow_angle)}, side={side_used})")

            rep_just_completed = (self.current_state == self.STATE_EXTENDING and new_state == self.STATE_EXTENDED)

            self.current_state = new_state
            self.prev_elbow_angle = elbow_angle

            if self.current_state == self.STATE_FLEXED:
                self.flexed_snapshot_errors = self._check_flexed_form(elbow_angle, trunk_angle)

            if rep_just_completed:
                self.session.record_rep(self.flexed_snapshot_errors)
                self.flexed_snapshot_errors = []

        return {
            "points": {"elbow": elbow, "shoulder": shoulder},
            "angles": {"elbow": elbow_angle, "trunk": trunk_angle},
        }


# -----------------------------
# 플랭크 판별 로직 (PlankChecker)
# 모델 호출 방식만 TRTModel로 교체, 나머지 로직은 동일
# -----------------------------
class PlankChecker:
    name = "plank"

    STATE_NOT_IN_PLANK = "NOT_IN_PLANK"
    STATE_HOLDING = "HOLDING"

    HORIZONTAL_TILT_THRESHOLD = 45

    ALIGNMENT_CLEAR_OK = 155
    ALIGNMENT_CLEAR_ERROR = 110
    MODEL_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, target_hold_seconds=60):
        self.session = ExerciseSession(target_reps=1)
        self.target_hold_seconds = target_hold_seconds
        self.current_state = self.STATE_NOT_IN_PLANK
        self.hold_start_time = None
        self.hold_errors_seen = set()

    def reset(self):
        self.current_state = self.STATE_NOT_IN_PLANK
        self.hold_start_time = None
        self.hold_errors_seen = set()
        self.session.reset()

    def _check_form(self, body_alignment_angle, hip_deviation):
        if body_alignment_angle >= self.ALIGNMENT_CLEAR_OK:
            return []

        if body_alignment_angle <= self.ALIGNMENT_CLEAR_ERROR:
            if hip_deviation > 0:
                return ["Raise your hips (back is too low)"]
            else:
                return ["Lower your hips (back is too high)"]

        # ---- TensorRT 추론 (기존 tensorflow model.predict 대체) ----
        input_array = np.array([body_alignment_angle, hip_deviation], dtype=np.float32)
        output = plank_model.predict(input_array)
        prob = float(output[0])

        if prob > self.MODEL_CONFIDENCE_THRESHOLD:
            if hip_deviation > 0:
                return ["Raise your hips (back is too low)"]
            else:
                return ["Lower your hips (back is too high)"]

        return []

    def _select_best_side(self, landmarks, w, h):
        l_shoulder, l_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_hip, l_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_ankle, l_ankle_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)
        left_visibility_sum = l_shoulder_vis + l_hip_vis + l_ankle_vis

        r_shoulder, r_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_hip, r_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_ankle, r_ankle_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)
        right_visibility_sum = r_shoulder_vis + r_hip_vis + r_ankle_vis

        if left_visibility_sum >= right_visibility_sum:
            shoulder, hip, ankle = l_shoulder, l_hip, l_ankle
            min_vis = min(l_shoulder_vis, l_hip_vis, l_ankle_vis)
            side_used = "left"
        else:
            shoulder, hip, ankle = r_shoulder, r_hip, r_ankle
            min_vis = min(r_shoulder_vis, r_hip_vis, r_ankle_vis)
            side_used = "right"

        return shoulder, hip, ankle, min_vis, side_used

    def update(self, landmarks, w, h):
        shoulder, hip, ankle, min_vis, side_used = self._select_best_side(landmarks, w, h)

        min_visibility = 0.5
        if min_vis <= min_visibility:
            return None

        body_tilt = calculate_horizontal_tilt(shoulder, ankle)
        is_horizontal = abs(body_tilt) < self.HORIZONTAL_TILT_THRESHOLD

        body_alignment_angle = calculate_angle(shoulder, hip, ankle)
        hip_deviation = calculate_hip_deviation(shoulder, hip, ankle)

        elapsed = 0.0

        if not self.session.set_completed:
            if self.current_state == self.STATE_NOT_IN_PLANK:
                if is_horizontal:
                    self.current_state = self.STATE_HOLDING
                    self.hold_start_time = time.time()
                    self.hold_errors_seen = set()
                    print("[STATE CHANGE] NOT_IN_PLANK -> HOLDING")

            elif self.current_state == self.STATE_HOLDING:
                if not is_horizontal:
                    print("[HOLD BROKEN] Plank position lost, resetting timer.")
                    self.current_state = self.STATE_NOT_IN_PLANK
                    self.hold_start_time = None
                    self.hold_errors_seen = set()
                else:
                    frame_errors = self._check_form(body_alignment_angle, hip_deviation)
                    self.hold_errors_seen.update(frame_errors)

                    elapsed = time.time() - self.hold_start_time

                    if elapsed >= self.target_hold_seconds:
                        self.session.record_rep(list(self.hold_errors_seen))
                        self.current_state = self.STATE_NOT_IN_PLANK
                        self.hold_start_time = None
                        self.hold_errors_seen = set()

        return {
            "points": {"hip": hip, "shoulder": shoulder},
            "angles": {"alignment": body_alignment_angle, "deviation": hip_deviation},
            "state": self.current_state,
            "elapsed": elapsed,
            "target_seconds": self.target_hold_seconds,
        }


# -----------------------------
# 운동 이름 -> Checker 매핑
# -----------------------------
def get_exercise_checker(exercise_name, target_reps=10):
    if exercise_name == "squat":
        return SquatChecker(target_reps)
    elif exercise_name == "neck_stretch":
        return NeckStretchChecker(target_reps)
    elif exercise_name == "bicep_curl":
        return BicepCurlChecker(target_reps)
    elif exercise_name == "plank":
        return PlankChecker(target_hold_seconds=TARGET_HOLD_SECONDS)
    else:
        raise ValueError(f"알 수 없는 운동: {exercise_name}")


# -----------------------------
# 운동별 화면 표시 함수
# -----------------------------
def draw_squat_angles(frame, result):
    knee = result["points"]["knee"]
    hip = result["points"]["hip"]
    angles = result["angles"]

    cv2.putText(frame, f'Knee: {int(angles["knee"])}',
                (int(knee[0]) + 10, int(knee[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f'Hip: {int(angles["hip"])}',
                (int(hip[0]) + 10, int(hip[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f'Trunk: {int(angles["trunk"])}',
                (int(hip[0]) + 10, int(hip[1]) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_neck_angles(frame, result):
    ref_point = result["points"]["nose_ref"]
    angles = result["angles"]

    cv2.putText(frame, f'Neck tilt: {int(angles["neck_tilt"])}',
                (int(ref_point[0]) + 10, int(ref_point[1]) - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_bicep_angles(frame, result):
    elbow = result["points"]["elbow"]
    shoulder = result["points"]["shoulder"]
    angles = result["angles"]

    cv2.putText(frame, f'Elbow: {int(angles["elbow"])}',
                (int(elbow[0]) + 10, int(elbow[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f'Trunk: {int(angles["trunk"])}',
                (int(shoulder[0]) + 10, int(shoulder[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_plank_status(frame, result):
    hip = result["points"]["hip"]
    angles = result["angles"]

    cv2.putText(frame, f'Align: {int(angles["alignment"])}',
                (int(hip[0]) + 10, int(hip[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f'Dev: {angles["deviation"]:.2f}',
                (int(hip[0]) + 10, int(hip[1]) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f'State: {result["state"]}',
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if result["state"] == "HOLDING":
        cv2.putText(frame, f'Hold: {result["elapsed"]:.1f}s / {result["target_seconds"]}s',
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


DRAW_FUNCTIONS = {
    "squat": draw_squat_angles,
    "neck_stretch": draw_neck_angles,
    "bicep_curl": draw_bicep_angles,
    "plank": draw_plank_status,
}


# -----------------------------
# 화면에 세션 상태(Reps, 피드백/요약) 그리는 함수
# -----------------------------
def draw_session_status(frame, session):
    cv2.putText(frame, f'Reps: {session.rep_count}/{session.target_reps}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if session.set_completed:
        y_offset = 80
        cv2.putText(frame, 'SET COMPLETE - Summary:', (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        y_offset += 35
        for err_name, count in session.error_counter.items():
            color = (0, 255, 0) if err_name == "Good form" else (0, 0, 255)
            cv2.putText(frame, f'{err_name}: {count}', (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            y_offset += 30
    elif session.has_completed_rep:
        if session.last_rep_errors:
            y_offset = 80
            for err in session.last_rep_errors:
                cv2.putText(frame, err, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                y_offset += 30
        else:
            cv2.putText(frame, 'Good form (last rep)', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Complete a rep to see feedback', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)


# -----------------------------
# 메인 루프
# -----------------------------
TARGET_REPS = 10
TARGET_HOLD_SECONDS = 60


def main(exercise_list=None):
    if exercise_list is None:
        exercise_list = ["squat"]

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 카메라 연결 상태나 장치 번호(0)를 확인하세요.")
        return

    checkers = [get_exercise_checker(name, target_reps=TARGET_REPS) for name in exercise_list]
    current_index = 0

    print(f"카메라 연결 성공. 운동 목록: {exercise_list}. 'r' 리셋, 'q' 종료.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        checker = checkers[current_index]
        draw_fn = DRAW_FUNCTIONS[checker.name]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            result = checker.update(landmarks, w, h)

            if result is not None:
                draw_fn(frame, result)
                draw_session_status(frame, checker.session)
            else:
                cv2.putText(frame, 'Low visibility - move into frame',
                            (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                draw_session_status(frame, checker.session)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        else:
            draw_session_status(frame, checker.session)
            cv2.putText(frame, 'No person detected',
                        (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f'Exercise: {checker.name} ({current_index + 1}/{len(checkers)})',
                    (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        if checker.session.set_completed and current_index < len(checkers) - 1:
            current_index += 1
            print(f"[EXERCISE CHANGE] Moving to: {checkers[current_index].name}")

        cv2.putText(frame, "Press 'r' to reset  |  Press 'q' to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Exercise Pose Test', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):
            checker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #main(["squat"])
    main(["bicep_curl"])
    #main(["plank"])
