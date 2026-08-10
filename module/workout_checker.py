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
    elif exercise_name == "bicep_curl":
        return BicepCurlChecker(target_reps)
    elif exercise_name == "plank":
        return PlankChecker(target_hold_seconds=TARGET_HOLD_SECONDS)
    else:
        raise ValueError(f"알 수 없는 운동: {exercise_name}")
