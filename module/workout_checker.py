"""
운동 자세 실시간 판별 모듈 (규칙 기반 버전)

스쿼트 / 푸시업 / 런지의 자세를 실시간으로 판별하고 rep(횟수)을 카운트한다.
바이셉컬·플랭크는 종목 구성이 스쿼트/푸시업/런지/noaction(LSTM 자동 인식)으로
바뀌면서 제거되었다. 모든 종목은 학습 모델 없이 순수 규칙(각도 임계값)으로만
정상/오류를 판별한다.

Mediapipe 초기화·좌표 추출·기본 각도 함수는 동현님 프로젝트의
mediapipe_op / basic_fn 모듈을 재사용한다.

TRTModel 클래스는 이후 LSTM(운동 종류 자동 인식) 엔진을 로드할 때
재사용할 예정이라 그대로 남겨둔다. (지금은 어떤 Checker도 모델을 쓰지 않음)
"""

import math
import time

import numpy as np

# ===== LSTM 연결 전까지는 미사용, 클래스만 남겨둠 =====
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ===== 동현님 프로젝트 함수 재사용 =====
# mp_pose        : PoseLandmark enum 접근용
# get_landmark_xy: 랜드마크 -> 픽셀 좌표 + visibility
from module.mediapipe_op import mp_pose, get_landmark_xy


# =========================================================
# TRT 추론 클래스 (현재 어떤 Checker도 사용하지 않음)
# LSTM(.trt) 엔진이 준비되면 여기서 로드해서 "운동 종류 인식" 용도로 사용 예정.
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

#  LSTM 엔진이 준비되면 여기서 다시 로드


# =========================================================
# 공통 각도 계산 함수
# (basic_fn.cal_angle_3point 과 동일 로직이지만, 이 모듈 내
#  세 점 각도는 아래 calculate_angle 로 통일해서 사용)
# =========================================================
def calculate_angle(a, b, c):
    """세 점 a-b-c 에서 b를 정점으로 하는 각도(도). 0~180 범위로 보정."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_angle_pushup(a, b, c):
    """
    calculate_angle과 동일하지만 180도 보정을 하지 않음.
    푸시업의 몸통 정렬(엉덩이 처짐/솟음) 판별처럼,
    180도를 기준으로 방향성을 구분해야 하는 경우에 사용.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
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

        self.resting = False  # check if it is break time
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
        self.count = 0  # count reset for next set

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
# 측면 전용, 순수 규칙 기반 (학습 모델 사용 안 함)
# -----------------------------
class SquatChecker:
    name = "squat"

    STATE_STANDING = "STANDING"
    STATE_DESCENDING = "DESCENDING"
    STATE_BOTTOM = "BOTTOM"
    STATE_ASCENDING = "ASCENDING"

    STANDING_THRESHOLD = 160
    BOTTOM_THRESHOLD = 130

    # 확실한 정상/오류 구간 (순수 규칙 판별 기준)
    KNEE_CLEAR_ERROR = 100    # 이 이상이면 확실히 깊이 부족
    TRUNK_CLEAR_ERROR = 40    # 이 이상이면 확실히 오류 (허리 굽음)
    HIP_CLEAR_ERROR = 90      # 이 이상이면 확실히 오류 (고관절 굴곡 부족)

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
        순수 규칙 기반 판별. 학습 모델 호출 없음.
        """
        errors = []

        if knee_angle >= self.KNEE_CLEAR_ERROR:
            errors.append("Squat deeper")
        if trunk_angle >= self.TRUNK_CLEAR_ERROR:
            errors.append("Straighten your back")
        if hip_flex_angle >= self.HIP_CLEAR_ERROR:
            errors.append("Bend at your hips more")

        return errors

    def _select_best_side(self, landmarks, w, h):
        """
        Select better side between left or right
        """
        l_shoulder, l_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_hip, l_hv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_knee, l_kv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        l_ankle, l_av = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)
        left_sum = l_sv + l_hv + l_kv + l_av

        r_shoulder, r_sv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_hip, r_hv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_knee, r_kv = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        r_ankle, r_av = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)
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

        knee_angle = calculate_angle(hip, knee, ankle)
        trunk_angle = calculate_trunk_angle(shoulder, hip)
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


# -----------------------------
# 푸시업 판별 로직 (PushupChecker)
# FormFit 원작자 로직(count_reps_and_feedback의 'pushup' 분기) 이식,
# 상태 이름만 프로젝트 컨벤션에 맞게 EXTENDED/FLEXED로 변경.
# -----------------------------
class PushupChecker:
    name = "pushup"

    STATE_EXTENDED = "EXTENDED"   # 팔이 펴진 상태
    STATE_FLEXED = "FLEXED"       # 팔이 굽혀진 상태 (카운트 시점)

    ELBOW_THRESHOLD = 160

    SAGGING_THRESHOLD = 190   # 이보다 크면 엉덩이 처짐
    PIKING_THRESHOLD = 160    # 이보다 작으면 엉덩이 솟음

    def __init__(self, target_reps=5, target_count=10, rest_seconds=30):
        self.session = ExerciseSession(target_reps, target_count, rest_seconds)
        self.current_state = None
        self.last_rep_errors = []

    def reset(self):
        self.current_state = None
        self.last_rep_errors = []
        self.session.reset()

    def update(self, landmarks, w, h):
        self.session.update_rest()

        if self.session.done or self.session.resting:
            return {
                "resting": self.session.resting,
                "rest_remaining": self.session.rest_remaining(),
                "done": self.session.done
            }

        l_shoulder, l_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_elbow, l_elbow_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, w, h)
        l_wrist, l_wrist_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
        l_hip, l_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_knee, l_knee_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)

        r_shoulder, r_shoulder_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_elbow, r_elbow_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        r_wrist, r_wrist_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        r_hip, r_hip_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_knee, r_knee_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)

        min_vis = min(l_shoulder_vis, l_elbow_vis, l_wrist_vis, l_hip_vis, l_knee_vis,
                       r_shoulder_vis, r_elbow_vis, r_wrist_vis, r_hip_vis, r_knee_vis)
        if min_vis <= 0.5:
            return None

        left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        left_body_angle = calculate_angle_pushup(l_shoulder, l_hip, l_knee)
        right_body_angle = calculate_angle_pushup(r_shoulder, r_hip, r_knee)

        if not self.session.done:
            rep_just_completed = False

            if (left_elbow_angle > self.ELBOW_THRESHOLD) and (right_elbow_angle > self.ELBOW_THRESHOLD) \
                    and self.current_state != self.STATE_EXTENDED:
                self.current_state = self.STATE_EXTENDED
            elif (left_elbow_angle < self.ELBOW_THRESHOLD) and (right_elbow_angle < self.ELBOW_THRESHOLD) \
                    and self.current_state == self.STATE_EXTENDED:
                self.current_state = self.STATE_FLEXED
                rep_just_completed = True

            errors = []
            if left_body_angle > self.SAGGING_THRESHOLD or right_body_angle > self.SAGGING_THRESHOLD:
                errors.append("Tighten core, no sagging hips")
            if left_body_angle < self.PIKING_THRESHOLD or right_body_angle < self.PIKING_THRESHOLD:
                errors.append("Lower hips to straight line")

            self.last_rep_errors = errors

            if rep_just_completed:
                self.session.record_count(errors)

        return {
            "points": {"elbow": l_elbow, "hip": l_hip},
            "angles": {"elbow": left_elbow_angle, "body": left_body_angle},
        }


# -----------------------------
# 런지 판별 로직 (LungeChecker)
# FormFit 원작자 로직(count_reps_and_feedback의 'lunge' 분기) 이식.
# front=left, back=right 고정 (원작자 코드와 동일한 전제).
# -----------------------------
class LungeChecker:
    name = "lunge"

    STATE_UP = "UP"
    STATE_DOWN = "DOWN"

    KNEE_THRESHOLD = 140
    SHALLOW_KNEE_ANGLE = 110    # front_knee_angle이 이보다 크면 얕은 런지로 판단

    def __init__(self, target_reps=5, target_count=10, rest_seconds=30):
        self.session = ExerciseSession(target_reps, target_count, rest_seconds)
        self.current_state = None
        self.last_rep_errors = []
        self.bottom_snapshot_errors = []

    def reset(self):
        self.current_state = None
        self.last_rep_errors = []
        self.bottom_snapshot_errors = []
        self.session.reset()

    def _check_form(self, front_knee_angle):
        errors = []
        if front_knee_angle > self.SHALLOW_KNEE_ANGLE:
            errors.append("Lunge deeper")
        return errors

    def update(self, landmarks, w, h):
        self.session.update_rest()

        if self.session.done or self.session.resting:
            return {
                "resting": self.session.resting,
                "rest_remaining": self.session.rest_remaining(),
                "done": self.session.done
            }

        front_shoulder, fs_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        front_hip, fh_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        front_knee, fk_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        front_ankle, fa_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)

        back_hip, bh_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        back_knee, bk_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        back_ankle, ba_vis = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)

        min_vis = min(fs_vis, fh_vis, fk_vis, fa_vis, bh_vis, bk_vis, ba_vis)
        if min_vis <= 0.5:
            return None

        front_knee_angle = calculate_angle(front_hip, front_knee, front_ankle)
        back_knee_angle = calculate_angle(back_hip, back_knee, back_ankle)

        if not self.session.done:
            rep_just_completed = False

            if front_knee_angle < self.KNEE_THRESHOLD and back_knee_angle > self.KNEE_THRESHOLD:
                self.current_state = self.STATE_DOWN
                self.bottom_snapshot_errors = self._check_form(front_knee_angle)

            if front_knee_angle > self.KNEE_THRESHOLD and self.current_state == self.STATE_DOWN:
                self.current_state = self.STATE_UP
                rep_just_completed = True

            if rep_just_completed:
                self.last_rep_errors = self.bottom_snapshot_errors
                self.session.record_count(self.bottom_snapshot_errors)
                self.bottom_snapshot_errors = []

        return {
            "points": {"knee": front_knee, "hip": front_hip},
            "angles": {"front_knee": front_knee_angle, "back_knee": back_knee_angle},
        }


# =========================================================
# 운동 이름 -> Checker 매핑
# =========================================================
def get_exercise_checker(exercise_name, target_reps=5, target_count=10, rest_seconds=30):
    if exercise_name == "squat":
        return SquatChecker(target_reps, target_count, rest_seconds)
    elif exercise_name == "pushup":
        return PushupChecker(target_reps, target_count, rest_seconds)
    elif exercise_name == "lunge":
        return LungeChecker(target_reps, target_count, rest_seconds)
    else:
        raise ValueError(f"알 수 없는 운동: {exercise_name}")
