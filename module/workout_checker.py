"""
운동 자세 실시간 판별 모듈 (규칙 기반 버전)

스쿼트 / 바이셉컬 / 플랭크의 자세를 실시간으로 판별하고 rep(횟수)을 카운트한다.
원래 팀원 코드는 바이셉컬·플랭크 폼 체크에 TensorRT(.trt) 모델을 사용했으나,
여기서는 TRT 관련 부분을 모두 주석 처리하고 각도 규칙만으로 판정한다.
(스쿼트는 원래 순수 규칙 기반이라 변경 없음)

MediaPipe 초기화·좌표 추출·기본 각도 함수는 동현님 프로젝트의
mediapipe_op / basic_fn 모듈을 재사용한다.
"""

import math
import time

import numpy as np

# ===== TRT 미사용 — 규칙 기반으로만 동작 =====
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ===== 동현님 프로젝트 함수 재사용 =====
# mp_pose        : PoseLandmark enum 접근용
# get_landmark_xy: 랜드마크 -> 픽셀 좌표 + visibility
from module.mediapipe_op import mp_pose, get_landmark_xy


# =========================================================
# TRT 추론 클래스 (미사용 — 전체 주석)
# .trt 엔진이 준비되면 아래 주석을 풀고, 각 Checker의
# _check_*_form 에서 규칙 기반 대신 모델 추론을 쓰면 된다.
# =========================================================
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
        input_array = np.ascontiguousarray(input_array, dtype=np.float32)
        output_array = np.empty(self.output_size, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, input_array, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_array, self.d_output, self.stream)
        self.stream.synchronize()
        return output_array

# ===== 학습된 TRT 엔진 로드 =====
bicep_curl_model = TRTModel("module/models/bicep_curl_classifier.trt")
plank_model      = TRTModel("module/models/plank_classifier.trt")
squat_model      = TRTModel("module/models/squat_classifier.trt")


# =========================================================
# 공통 각도 계산 함수
# (basic_fn.cal_angle_3point 과 동일 로직이지만, 이 모듈 내
#  세 점 각도는 아래 calculate_angle 로 통일해서 사용)
# =========================================================
def calculate_angle(a, b, c):
    """세 점 a-b-c 에서 b를 정점으로 하는 각도(도)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_trunk_angle(shoulder, hip):
    """어깨-엉덩이 선이 수직에서 벗어난 정도(도). 0에 가까울수록 곧게 섬."""
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def calculate_horizontal_tilt(point_a, point_b):
    """두 점을 잇는 선의 수평 기준 기울기(도). 0에 가까우면 수평. -90~90."""
    dx = point_b[0] - point_a[0]
    dy = -(point_b[1] - point_a[1])  # 화면 좌표는 아래가 +y이므로 반전
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def calculate_hip_deviation(shoulder, hip, ankle):
    """
    어깨-발목 직선 대비 엉덩이가 위/아래로 벗어난 정도.
    양수: 엉덩이가 처짐(back too low) / 음수: 엉덩이가 솟음(back too high).
    """
    if ankle[0] == shoulder[0]:
        expected_hip_y = (shoulder[1] + ankle[1]) / 2
    else:
        t = (hip[0] - shoulder[0]) / (ankle[0] - shoulder[0])
        expected_hip_y = shoulder[1] + t * (ankle[1] - shoulder[1])
    return hip[1] - expected_hip_y


# =========================================================
# 운동 세션 관리 (모든 운동 공통)
# =========================================================
class ExerciseSession:
    def __init__(self, target_reps, target_count, rest_seconds=30):
        self.target_reps = target_reps
        self.target_count = target_count
        self.rest_seconds = rest_seconds
        self.reset()

    def reset(self):
        self.rep_count = 0
        self.count = 0
        self.error_counter = {}
        self.last_rep_errors = []
        self.has_completed_rep = False

        self.resting = False # check if it is break time
        self.rest_start = None
        self.done = False
        print("[RESET] Session restarted.")

    def record_count(self, errors):
        """Call at every move"""
        # Check if it is break time
        # if it is break time, ignore it
        if self.resting or self.done:
            return

        self.count += 1

        if errors:
            for err in errors:
                self.error_counter[err] = self.error_counter.get(err, 0) + 1
        else:
            self.error_counter["Good form"] = self.error_counter.get("Good form", 0) + 1

        print(f"[COUNT] {self.count}/{self.target_count}  "
              f"(set {self.rep_count + 1}/{self.target_reps})")

        # If done one set, make rep complete
        if self.count >= self.target_count:
            self._complete_set()

    def _complete_set(self):
        self.rep_count += 1
        self.count = 0 # count reset for next set

        print("-" * 40)
        print(f"[SET DONE] {self.rep_count}/{self.target_reps} set complete.")

        # Total complete when every set is over
        if self.rep_count >= self.target_reps:
            self.done = True
            print("=" * 40)
            print(f"[EXERCISE DONE] {self.target_reps} sets finished.")
            for err_name, count in self.error_counter.items():
                print(f"  - {err_name}: {count} time(s)")
            print("=" * 40)

        else:
            self.resting = True
            self.rest_start = time.time()
            print(f"[REST] {self.rest_seconds}s rest started.")

    def update_rest(self):
        """
        Call every frame
        Check if break is over and prepare for the next set
        """
        if not self.resting:
            return
        elapsed = time.time() - self.rest_start
        if elapsed >= self.rest_seconds:
            self.resting = False
            self.rest_start = None
            print("[REST DONE] Next set start.")

    def rest_remaining(self):
        """
        Remain resting time (sec)
        If not resting, 0
        """
        if not self.resting:
            return 0.0
        return max(0.0, self.rest_seconds - (time.time() - self.rest_start))

# -----------------------------
# 스쿼트 판별 로직 (SquatChecker)
# 측면 전용 규칙 기반 판별 (기존 로직 그대로 유지)
# -----------------------------
class SquatChecker:
    name = "squat"

    STATE_STANDING   = "STANDING"
    STATE_DESCENDING = "DESCENDING"
    STATE_BOTTOM     = "BOTTOM"
    STATE_ASCENDING  = "ASCENDING"

    STANDING_THRESHOLD = 160
    BOTTOM_THRESHOLD   = 110

    # 확실한 정상/오류 구간 (하이브리드 판별 기준)
    KNEE_CLEAR_OK      = 100    # 이 이하면 확실히 충분한 깊이
    KNEE_CLEAR_ERROR   = 130    # 이 이상이면 확실히 깊이 부족
    TRUNK_CLEAR_OK     = 30     # 이 이하면 확실히 정상 (허리 안 굽음)
    TRUNK_CLEAR_ERROR  = 40     # 이 이상이면 확실히 오류 (허리 굽음)
    HIP_CLEAR_ERROR    = 100    # 이 이상이면 확실히 오류 (고관절 굴곡 부족)

    MODEL_CONFIDENCE_THRESHOLD = 0.5


    def __init__(self, target_reps=5, target_count=10, rest_seconds=30):
        self.session = ExerciseSession(target_reps, target_count, rest_seconds)
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
        """
        규칙+모델 하이브리드 판별.
        확실한 정상/오류 구간은 모델 호출 없이 바로 처리하고,
        애매한 구간에서만 학습된 모델을 호출.
        오류 유형(깊이부족/허리굽음/고관절부족)은 항상 규칙이 결정한다.
        """
        errors = []

        # 1. 규칙: 확실한 오류 (모델 호출 없이 즉시 판정)
        if knee_angle >= self.KNEE_CLEAR_ERROR:
            errors.append("Squat deeper")
        if trunk_angle >= self.TRUNK_CLEAR_ERROR:
            errors.append("Straighten your back")
        if hip_flex_angle >= self.HIP_CLEAR_ERROR:
            errors.append("Bend at your hips more")

        if errors:
            return errors

        # 2. 규칙: 확실한 정상 (모델 호출 없이 즉시 통과) 
        if knee_angle <= self.KNEE_CLEAR_OK and trunk_angle <= self.TRUNK_CLEAR_OK:
            return []

        # 3. 애매한 구간: 학습된 모델(TensorRT)로 최종 판정 
        input_array = np.array([knee_angle, hip_flex_angle, trunk_angle], dtype=np.float32)
        output = squat_model.predict(input_array)
        prob = float(output[0])

        if prob > self.MODEL_CONFIDENCE_THRESHOLD:
            # 모델은 "오류"만 판단, 어떤 오류인지는 규칙이 다시 결정
            if knee_angle > self.KNEE_CLEAR_OK:
                errors.append("Squat deeper")
            if trunk_angle > self.TRUNK_CLEAR_OK:
                errors.append("Straighten your back")
            if hip_flex_angle > 90:
                errors.append("Bend at your hips more")
            if not errors:
                errors.append("Check your form")

        return errors

    def _select_best_side(self, landmarks, w, h):
        """
        Select better side between left or right
        """
        l_shoulder, l_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_hip, l_hv      = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_knee, l_kv     = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        l_ankle, l_av    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)
        left_sum = l_sv + l_hv + l_kv + l_av

        r_shoulder, r_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_hip, r_hv      = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_knee, r_kv     = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        r_ankle, r_av    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)
        right_sum = r_sv + r_hv + r_kv + r_av

        if left_sum >= right_sum:
            return l_shoulder, l_hip, l_knee, l_ankle, min(l_sv, l_hv, l_kv, l_av), "left"
        else:
            return r_shoulder, r_hip, r_knee, r_ankle, min(r_sv, r_hv, r_kv, r_av), "right"

    def update(self, landmarks, w, h):
        # check if it is resting time
        self.session.update_rest()

        # if done or resting, skip checker
        if self.session.done or self.session.resting:
            return {
                "resting": self.session.resting,
                "rest_remaining": self.session.rest_remaining(),
                "done": self.session.done
            }

        # ----- Extract coordinate & state transition -----
        shoulder, hip, knee, ankle, min_vis, side_used = self._select_best_side(landmarks, w, h)

        if min_vis <= 0.5:
            return None

        knee_angle     = calculate_angle(hip, knee, ankle)
        trunk_angle    = calculate_trunk_angle(shoulder, hip)
        hip_flex_angle = calculate_angle(shoulder, hip, knee)

        if not self.session.done:
            new_state = self._update_state(knee_angle)

            if new_state != self.current_state:
                print(f"[STATE CHANGE] {self.current_state} -> {new_state}  "
                      f"(knee={int(knee_angle)}, side={side_used})")

            move_done = (self.current_state == self.STATE_ASCENDING and new_state == self.STATE_STANDING)

            self.current_state = new_state
            self.prev_knee_angle = knee_angle

            if self.current_state == self.STATE_BOTTOM:
                self.bottom_snapshot_errors = self._check_bottom_form(trunk_angle, knee_angle, hip_flex_angle)

            if move_done:
                self.session.record_count(self.bottom_snapshot_errors)
                self.bottom_snapshot_errors = []

        return {
            "points": {"knee": knee, "hip": hip},
            "angles": {"knee": knee_angle, "hip": hip_flex_angle, "trunk": trunk_angle},
        }


# =========================================================
# 바이셉컬 판별
# =========================================================
class BicepCurlChecker:
    name = "biceps_curl"   # 이름 통일: biceps_curl

    STATE_EXTENDED  = "EXTENDED"
    STATE_CURLING   = "CURLING"
    STATE_FLEXED    = "FLEXED"
    STATE_EXTENDING = "EXTENDING"

    EXTENDED_THRESHOLD = 150
    FLEXED_THRESHOLD   = 65

    TRUNK_CLEAR_OK = 8
    MODEL_CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, target_reps=10, target_count=10, rest_seconds=60):
        self.session = ExerciseSession(target_reps, target_count, rest_seconds)
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
        """
        Rule + Model
        check based on rule first
        after that, check with model to determine more correct
        """
        errors = []
        if trunk_angle <= self.TRUNK_CLEAR_OK:
            return errors

        # ----- TRT inference -----
        input_array = np.array([elbow_angle, trunk_angle], dtype=np.float32)
        output = bicep_curl_model.predict(input_array)
        prob = float(output[0])

        if prob > self.MODEL_CONFIDENCE_THRESHOLD:
            errors.append("Straighten your back (lean back detected)")
        return errors


    def _select_best_side(self, landmarks, w, h):
        l_shoulder, l_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_elbow, l_ev    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, w, h)
        l_wrist, l_wv    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
        l_hip, l_hv      = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        left_sum = l_sv + l_ev + l_wv + l_hv

        r_shoulder, r_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_elbow, r_ev    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        r_wrist, r_wv    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        r_hip, r_hv      = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        right_sum = r_sv + r_ev + r_wv + r_hv

        if left_sum >= right_sum:
            return l_shoulder, l_elbow, l_wrist, l_hip, min(l_sv, l_ev, l_wv, l_hv), "left"
        return r_shoulder, r_elbow, r_wrist, r_hip, min(r_sv, r_ev, r_wv, r_hv), "right"

    def update(self, landmarks, w, h):
        # check if it is resting time
        self.session.update_rest()

        # if done or resting, skip checker
        if self.session.done or self.session.resting:
            return {
                "resting": self.session.resting,
                "rest_remaining": self.session.rest_remaining(),
                "done": self.session.done
            }

        shoulder, elbow, wrist, hip, min_vis, side_used = self._select_best_side(landmarks, w, h)

        if min_vis <= 0.5:
            return None

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        trunk_angle = calculate_trunk_angle(shoulder, hip)

        if not self.session.done:
            new_state = self._update_state(elbow_angle)

            if new_state != self.current_state:
                print(f"[STATE CHANGE] {self.current_state} -> {new_state}  "
                      f"(elbow={int(elbow_angle)}, side={side_used})")

            move_done = (self.current_state == self.STATE_EXTENDING and new_state == self.STATE_EXTENDED)

            self.current_state = new_state
            self.prev_elbow_angle = elbow_angle

            if self.current_state == self.STATE_FLEXED:
                self.flexed_snapshot_errors = self._check_flexed_form(elbow_angle, trunk_angle)

            if move_done:
                self.session.record_count(self.flexed_snapshot_errors)
                self.flexed_snapshot_errors = []

        return {
            "points": {"elbow": elbow, "shoulder": shoulder},
            "angles": {"elbow": elbow_angle, "trunk": trunk_angle},
        }


# =========================================================
# 플랭크 판별 (TRT 제거 → 규칙 기반)
# =========================================================
class PlankChecker:
    name = "plank"

    STATE_NOT_IN_PLANK = "NOT_IN_PLANK"
    STATE_HOLDING      = "HOLDING"

    HORIZONTAL_TILT_THRESHOLD  = 45
    ALIGNMENT_CLEAR_OK         = 155
    ALIGNMENT_CLEAR_ERROR      = 110
    MODEL_CONFIDENCE_THRESHOLD = 0.5   # TRT 미사용

    def __init__(self, target_reps=3, hold_seconds=60, rest_seconds=30):
        # 플랭크는 "hold_seconds초 유지 = 1회"라 target_count=1로 고정
        self.session = ExerciseSession(target_reps, target_count=1, rest_seconds=rest_seconds)
        self.hold_seconds = hold_seconds       # 유지할 시간(초)
        self.current_state = self.STATE_NOT_IN_PLANK
        self.hold_start_time = None
        self.hold_errors_seen = set()

    def reset(self):
        self.current_state = self.STATE_NOT_IN_PLANK
        self.hold_start_time = None
        self.hold_errors_seen = set()
        self.session.reset()

    def _check_form(self, body_alignment_angle, hip_deviation):
        """TRT 미사용: 정렬 각도와 힙 편차 부호로 판정."""
        if body_alignment_angle >= self.ALIGNMENT_CLEAR_OK:
            return []

        if body_alignment_angle <= self.ALIGNMENT_CLEAR_ERROR:
            if hip_deviation > 0:
                return ["Raise your hips (back is too low)"]
            return ["Lower your hips (back is too high)"]

        # ---- TRT Inference in mid range -----
        input_array = np.array([body_alignment_angle, hip_deviation], dtype=np.float32)
        output = plank_model.predict(input_array)
        prob = float(output[0])
        if prob > self.MODEL_CONFIDENCE_THRESHOLD:
            # determined to error
            if hip_deviation > 0:
                return ["Raise your hips (back is too low)"]
            else:
                return ["Lower your hips (back is too high)"]
        return []


    def _select_best_side(self, landmarks, w, h):
        l_shoulder, l_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_hip, l_hv      = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_ankle, l_av    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)
        left_sum = l_sv + l_hv + l_av

        r_shoulder, r_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_hip, r_hv      = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_ankle, r_av    = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)
        right_sum = r_sv + r_hv + r_av

        if left_sum >= right_sum:
            return l_shoulder, l_hip, l_ankle, min(l_sv, l_hv, l_av), "left"
        return r_shoulder, r_hip, r_ankle, min(r_sv, r_hv, r_av), "right"

    def update(self, landmarks, w, h):
        self.session.update_rest()
        if self.session.done or self.session.resting:
            return {
                "resting": self.session.resting,
                "rest_remaining": self.session.rest_remaining(),
                "done": self.session.done,
            }

        shoulder, hip, ankle, min_vis, side_used = self._select_best_side(landmarks, w, h)

        if min_vis <= 0.5:
            return None

        body_tilt = calculate_horizontal_tilt(shoulder, ankle)
        is_horizontal = abs(body_tilt) < self.HORIZONTAL_TILT_THRESHOLD

        body_alignment_angle = calculate_angle(shoulder, hip, ankle)
        hip_deviation = calculate_hip_deviation(shoulder, hip, ankle)

        elapsed = 0.0

        if not self.session.done:
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
                    if elapsed >= self.hold_seconds:              # target_count → hold_seconds
                        self.session.record_count(list(self.hold_errors_seen))
                        self.current_state = self.STATE_NOT_IN_PLANK
                        self.hold_start_time = None
                        self.hold_errors_seen = set()

        return {
            "points": {"hip": hip, "shoulder": shoulder},
            "angles": {"alignment": body_alignment_angle, "deviation": hip_deviation},
            "state": self.current_state,
            "elapsed": elapsed,
            "target_seconds": self.hold_seconds,
        }


# =========================================================
# 운동 이름 -> Checker 매핑
# =========================================================
def get_exercise_checker(exercise_name, target_reps=5, target_count=10, rest_seconds=30):
    if exercise_name == "squat":
        return SquatChecker(target_reps, target_count, rest_seconds)
    elif exercise_name == "biceps_curl":          # 이름 통일
        return BicepCurlChecker(target_reps, target_count, rest_seconds)
    elif exercise_name == "plank":
        return PlankChecker(target_reps=target_reps, hold_seconds=target_count, rest_seconds=rest_seconds)
    else:
        raise ValueError(f"알 수 없는 운동: {exercise_name}")