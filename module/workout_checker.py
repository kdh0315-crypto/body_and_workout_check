"""
운동 자세 실시간 판별 모듈 (규칙 기반 + LSTM 운동 종류 자동 인식)

스쿼트 / 푸시업 / 런지의 자세를 실시간으로 판별하고 rep(횟수)을 카운트한다.
바이셉컬·플랭크는 종목 구성이 스쿼트/푸시업/런지/noaction(LSTM 자동 인식)으로
바뀌면서 제거되었다. 각 Checker의 정상/오류 판별은 학습 모델 없이 순수 규칙
(각도 임계값)으로만 이루어진다.

Mediapipe 초기화·좌표 추출·기본 각도 함수는 동현님 프로젝트의
mediapipe_op / basic_fn 모듈을 재사용한다.

TRTModel은 범용 TRT 추론 래퍼이고, ActionRecognizer가 이를 이용해
15프레임 시퀀스(132차원 = 33 landmark x [x,y,z,visibility])로부터
현재 운동 종류(pushup/squat/lunge/noactions)를 예측한다.
"""

import math
import time

import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ===== 동현님 프로젝트 함수 재사용 =====
# mp_pose        : PoseLandmark enum 접근용
# get_landmark_xy: 랜드마크 -> 픽셀 좌표 + visibility
from module.mediapipe_op import mp_pose, get_landmark_xy


# =========================================================
# TRT 추론 클래스 (범용 래퍼)
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


# =========================================================
# LSTM 기반 운동 종류 자동 인식 (ActionRecognizer)
# 최근 SEQ_LEN 프레임의 키포인트 시퀀스를 모아 TRTModel(lstm.trt)로 추론한다.
# =========================================================
# Engine: LSTM
# 10프레임 시퀀스로 학습한 엔진
# - 기존 30프레임보다 버퍼가 1/3이라 더 빠르게 반응이 가능
# - Jetson은 frame이 부족하여 이러한 모델이 더 정확한 판단 가능
LSTM_ENGINE_PATH = "models/lstm_10frame.trt"

ACTIONS = np.array(["pushup", "squat", "lunge", "noactions"])
SEQ_LEN = 10
NUM_FEATURES = 132   # 33 landmark x (x, y, z, visibility)


def extract_keypoints(landmarks):
    """PoseLandmarker 랜드마크 리스트 -> 132차원 벡터. 감지 실패 시 0으로."""
    if landmarks is None:
        return np.zeros(NUM_FEATURES, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
        dtype=np.float32,
    ).flatten()


class ActionRecognizer:
    """
    매 프레임 update(landmarks)를 호출해 키포인트를 쌓다가,
    SEQ_LEN 프레임이 모이면 LSTM 엔진으로 운동 종류를 예측한다.

    슬라이딩 윈도우 특성상 프레임마다 확률이 조금씩 흔들리는데, 클래스별 확률에
    EMA(지수이동평균)를 적용해 그 흔들림을 눌러준다 (레퍼런스의 최근 10프레임
    다수결과 같은 목적, EMA로 더 가볍게 구현).
    """

    def __init__(self, engine_path=LSTM_ENGINE_PATH, seq_len=SEQ_LEN,
                 num_features=NUM_FEATURES, actions=ACTIONS, threshold=0.5,
                 ema_alpha=0.5):
        self.model = TRTModel(engine_path)
        self.seq_len = seq_len
        self.num_features = num_features
        self.actions = actions
        self.threshold = threshold
        # alpha=1.0 -> 스무딩 없음 (윈도우 레퍼런스와 동일하게 매 프레임 값 그대로 사용).
        # 값을 낮추면(예: 0.3) 과거 프레임과 섞어서 흔들림을 눌러주는 대신 반응이 느려짐.
        self.ema_alpha = ema_alpha
        self.sequence = []
        self.ema_probs = None

    def reset(self):
        self.sequence = []
        self.ema_probs = None

    def update(self, landmarks):
        """
        landmarks: extract_landmarks_video 결과의 results.pose_landmarks[0] (또는 None)
        반환: SEQ_LEN 프레임이 아직 안 모였으면 None,
              모였으면 (action_name, confidence) — threshold 미만이면 action_name="..."
        """
        self.sequence.append(extract_keypoints(landmarks))
        self.sequence = self.sequence[-self.seq_len:]

        if len(self.sequence) < self.seq_len:
            return None

        probs = self.model.predict(np.array(self.sequence))   # (SEQ_LEN, NUM_FEATURES) -> (len(actions),)

        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs

        idx = int(np.argmax(self.ema_probs))
        confidence = float(self.ema_probs[idx])
        action_name = self.actions[idx] if confidence > self.threshold else "..."
        return action_name, confidence


def update_with_recognition(recognizer, checker, landmarks, w, h):
    """
    ActionRecognizer + Checker를 묶어 한 프레임을 처리하는 공용 헬퍼.
    LSTM이 인식한 운동이 checker.name과 같을 때만 checker.update()를 호출한다
    (test_sel_checker.py에서 검증된 것과 동일한 방식). GUI 등 다른 소비자가
    "인식 -> 일치할 때만 카운트" 흐름을 매번 새로 구현하지 않고 재사용할 수 있다.

    반환: (recognized_action, confidence, check_result)
      - landmarks가 None이면 셋 다 None.
      - 시퀀스 버퍼가 덜 찼으면 recognized_action/confidence가 None.
      - 인식된 운동이 checker.name과 다르면 check_result가 None
        (이 프레임은 카운트되지 않았다는 뜻).
    """
    if landmarks is None:
        return None, None, None

    recognized_action, confidence = None, None
    action_result = recognizer.update(landmarks)
    if action_result is not None:
        recognized_action, confidence = action_result

    check_result = None
    if recognized_action == checker.name:
        check_result = checker.update(landmarks, w, h)

    return recognized_action, confidence, check_result


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
        self.last_rep_errors = list(errors)

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
    STATE_BOTTOM = "BOTTOM"

    # 레퍼런스(FormFit)와 동일: 무릎각·고관절각 모두 좌우 동일 임계값(160)으로 왕복 판정
    THRESHOLD = 160

    # 확실한 정상/오류 구간 (순수 규칙 판별 기준)
    KNEE_CLEAR_ERROR = 100    # 이 이상이면 확실히 깊이 부족
    TRUNK_CLEAR_ERROR = 40    # 이 이상이면 확실히 오류 (허리 굽음)
    HIP_CLEAR_ERROR = 90      # 이 이상이면 확실히 오류 (고관절 굴곡 부족)

    def __init__(self, target_reps=5, target_count=10, rest_seconds=30):
        self.session = ExerciseSession(target_reps, target_count, rest_seconds)
        self.current_state = self.STATE_STANDING
        self.bottom_snapshot_errors = []

    def reset(self):
        self.current_state = self.STATE_STANDING
        self.bottom_snapshot_errors = []
        self.session.reset()

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

        # ----- Extract both-side coordinates (레퍼런스처럼 좌우 모두 사용) -----
        l_shoulder, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_hip, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_knee, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        l_ankle, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)

        r_shoulder, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_hip, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_knee, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        r_ankle, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)

        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

        if not self.session.done:
            # 레퍼런스와 동일한 두 개의 독립된 조건문 (좌우 무릎각·고관절각 모두 만족해야 함)
            went_down = (left_knee_angle < self.THRESHOLD and right_knee_angle < self.THRESHOLD
                         and left_hip_angle < self.THRESHOLD and right_hip_angle < self.THRESHOLD)
            came_up = (left_knee_angle > self.THRESHOLD and right_knee_angle > self.THRESHOLD
                       and left_hip_angle > self.THRESHOLD and right_hip_angle > self.THRESHOLD)

            if went_down and self.current_state != self.STATE_BOTTOM:
                trunk_angle = calculate_trunk_angle(l_shoulder, l_hip)
                self.bottom_snapshot_errors = self._check_bottom_form(
                    trunk_angle, left_knee_angle, left_hip_angle)
                self.current_state = self.STATE_BOTTOM
                print(f"[STATE CHANGE] {self.STATE_STANDING} -> {self.STATE_BOTTOM}  "
                      f"(knee L={int(left_knee_angle)} R={int(right_knee_angle)})")

            if came_up and self.current_state == self.STATE_BOTTOM:
                self.current_state = self.STATE_STANDING
                print(f"[STATE CHANGE] {self.STATE_BOTTOM} -> {self.STATE_STANDING}  "
                      f"(knee L={int(left_knee_angle)} R={int(right_knee_angle)})")
                self.session.record_count(self.bottom_snapshot_errors)
                self.bottom_snapshot_errors = []

        return {
            "points": {"knee": l_knee, "hip": l_hip},
            "angles": {"left_knee": left_knee_angle, "right_knee": right_knee_angle,
                       "left_hip": left_hip_angle, "right_hip": right_hip_angle},
            "state": self.current_state,
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

    ELBOW_THRESHOLD = 130

    SAGGING_THRESHOLD = 190   # 이보다 크면 엉덩이 처짐
    PIKING_THRESHOLD = 160    # 이보다 작으면 엉덩이 솟음

    def __init__(self, target_reps=5, target_count=10, rest_seconds=30):
        self.session = ExerciseSession(target_reps, target_count, rest_seconds)
        self.current_state = None

    def reset(self):
        self.current_state = None
        self.session.reset()

    def update(self, landmarks, w, h):
        self.session.update_rest()

        if self.session.done or self.session.resting:
            return {
                "resting": self.session.resting,
                "rest_remaining": self.session.rest_remaining(),
                "done": self.session.done
            }

        l_shoulder, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_elbow, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, w, h)
        l_wrist, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
        l_hip, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_knee, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)

        r_shoulder, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_elbow, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        r_wrist, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        r_hip, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_knee, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)

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

            if rep_just_completed:
                self.session.record_count(errors)

        return {
            "points": {"elbow": l_elbow, "hip": l_hip},
            "angles": {
                "left_elbow": left_elbow_angle, "right_elbow": right_elbow_angle,
                "left_body": left_body_angle, "right_body": right_body_angle,
            },
            "state": self.current_state,
        }


# -----------------------------
# 런지 판별 로직 (LungeChecker)
# FormFit 원작자 로직(count_reps_and_feedback의 'lunge' 분기) 이식.
# 원작은 front=left, back=right로 고정이었으나, 어느 쪽 다리로 내딛어도 잡히도록
# 매 프레임 더 굽혀진(각도가 작은) 쪽을 앞다리로 판단하는 방식으로 바꿨다.
# -----------------------------
class LungeChecker:
    name = "lunge"

    STATE_UP = "UP"
    STATE_DOWN = "DOWN"

    KNEE_THRESHOLD     = 140
    SHALLOW_KNEE_ANGLE = 110    # front_knee_angle이 이보다 크면 얕은 런지로 판단

    def __init__(self, target_reps=5, target_count=10, rest_seconds=30):
        self.session = ExerciseSession(target_reps, target_count, rest_seconds)
        self.current_state = None
        self.bottom_snapshot_errors = []

    def reset(self):
        self.current_state = None
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

        l_hip, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        l_knee, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        l_ankle, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)

        r_hip, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        r_knee, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        r_ankle, _ = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)

        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # 더 굽혀진(각도가 작은) 쪽이 앞다리 — 왼쪽/오른쪽 어느 쪽으로 내딛어도 대응
        if left_knee_angle <= right_knee_angle:
            front_knee, front_hip = l_knee, l_hip
            front_knee_angle, back_knee_angle = left_knee_angle, right_knee_angle
        else:
            front_knee, front_hip = r_knee, r_hip
            front_knee_angle, back_knee_angle = right_knee_angle, left_knee_angle

        if not self.session.done:
            rep_just_completed = False

            if front_knee_angle < self.KNEE_THRESHOLD and back_knee_angle < self.KNEE_THRESHOLD:
                self.current_state = self.STATE_DOWN
                self.bottom_snapshot_errors = self._check_form(front_knee_angle)

            if front_knee_angle > self.KNEE_THRESHOLD and self.current_state == self.STATE_DOWN:
                self.current_state = self.STATE_UP
                rep_just_completed = True

            if rep_just_completed:
                self.session.record_count(self.bottom_snapshot_errors)
                self.bottom_snapshot_errors = []

        return {
            "points": {"knee": front_knee, "hip": front_hip},
            "angles": {"front_knee": front_knee_angle, "back_knee": back_knee_angle},
            "state": self.current_state,
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
