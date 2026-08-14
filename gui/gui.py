import sys
import time
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtWidgets import QProgressBar, QTextEdit

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QComboBox, QGroupBox, QRadioButton, QButtonGroup
)

from module.mediapipe_op import *      # CaptureForm, save_with_pose 등
from module.workout_sel import *
from module.cal_angle import *
from module.ollama_op import *
from module.workout_checker import get_exercise_checker, ActionRecognizer, update_with_recognition


from gui.gui_style import *


class LLMWorker(QThread):
    """스트레칭 추천 + 운동 처방을 백그라운드에서 순서대로 실행."""
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, metrics, user_info):
        super().__init__()
        self.metrics = metrics
        self.user_info = user_info

    def run(self):
        abnormal = find_abnormal(self.metrics)

        stretches = recommend_stretches(abnormal)

        goal = self.user_info.get("goal", "posture")
        level = self.user_info.get("level", "beginner")
        age = self.user_info.get("age")
        workout = prescribe_workouts(goal, level, age)

        if not stretches and not workout.get("recommendations"):
            self.failed.emit("추천을 생성하지 못했습니다. 다시 시도해주세요.")
            return

        self.finished.emit({
            "stretches": stretches,
            "workout": workout,
        })


class LoadingOverlay(QWidget):
    """화면 위에 덮는 로딩 표시."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(13, 17, 23, 235);")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        self.label = QLabel("자세 분석 중...\n잠시만 기다려주세요")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "color: #e6edf3; font-size: 18px; font-weight: 800; background: transparent;"
        )
        layout.addWidget(self.label)

        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedWidth(240)
        layout.addWidget(bar, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def resizeEvent(self, event):
        self.resize(self.parent().size())


class CameraView(QWidget):
    def __init__(self, user_info: dict):
        super().__init__()
        self.setWindowTitle("자세 촬영")
        self.setMinimumWidth(560)
        self.user_info = user_info
        self.cap_form = CaptureForm()

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(10)

        title = QLabel("자세 촬영")
        title.setObjectName("title")
        layout.addWidget(title)

        self.status_label = QLabel("정면 자세를 잡고 '촬영' 버튼을 누르세요")
        self.status_label.setObjectName("subtitle")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        video_card = QWidget()
        video_card.setObjectName("cardFlat")
        video_card_layout = QVBoxLayout()
        video_card_layout.setContentsMargins(14, 14, 14, 14)

        self.video_label = QLabel("카메라 준비 중...")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        video_card_layout.addWidget(self.video_label)

        video_card.setLayout(video_card_layout)
        layout.addWidget(video_card)

        self.result_view = QTextEdit()
        self.result_view.setObjectName("resultView")
        self.result_view.setReadOnly(True)
        self.result_view.setMinimumHeight(200)
        self.result_view.hide()
        layout.addWidget(self.result_view)

        self.capture_btn = QPushButton("촬영하기")
        self.capture_btn.setObjectName("primary")
        self.capture_btn.clicked.connect(self.on_capture)
        layout.addWidget(self.capture_btn)

        self.start_workout_btn = QPushButton("운동 시작하기")
        self.start_workout_btn.setObjectName("primary")
        self.start_workout_btn.clicked.connect(self.on_start_workout)
        self.start_workout_btn.hide()
        layout.addWidget(self.start_workout_btn)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_frame = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.selector = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def on_capture(self):
        if self.current_frame is None:
            return

        self.cap_form.key_released = True
        result = self.cap_form.capture_form(ord('c'), self.current_frame)

        if not self.cap_form.front_img_en:
            self.status_label.setText("정면 자세를 잡고 '촬영'을 누르세요")
        elif not self.cap_form.side_img_en:
            self.status_label.setText("측면 자세를 잡고 '촬영'을 누르세요")

        if result is not None:
            front_img, side_img = result
            self.status_label.setText("촬영 완료 — 분석 중...")
            cv2.imwrite("front_raw.jpg", front_img)
            cv2.imwrite("side_raw.jpg", side_img)
            self.timer.stop()
            self.on_capture_done(front_img, side_img)

    def on_capture_done(self, front_img, side_img):
        print("촬영 완료. user_info:", self.user_info)

        self.timer.stop()
        self.video_label.clear()
        self.video_label.setText("")

        front_results = save_with_pose(front_img, "front_pose.jpg")
        side_results = save_with_pose(side_img, "side_pose.jpg")

        if not has_landmarks(front_results) or not has_landmarks(side_results):
            self.status_label.setText("포즈 감지 실패 — 다시 촬영")
            self._restart()
            return

        fh, fw = front_img.shape[:2]
        sh, sw = side_img.shape[:2]
        front_features = calculate_all_features(front_results.pose_landmarks[0], fw, fh)
        side_features = calculate_all_features(side_results.pose_landmarks[0], sw, sh)

        print("FHA Rule:", side_features.get("fha_deg"), "/", classify_fha(side_features.get("fha_deg")))
        print("front_features:", front_features)
        print("side_features:", side_features)

        metrics = features_to_metrics(front_features, side_features)
        print("metrics:", metrics)

        self.overlay = LoadingOverlay(self)
        self.overlay.resize(self.size())
        self.overlay.show()

        self.worker = LLMWorker(metrics, self.user_info)
        self.worker.finished.connect(self.on_llm_done)
        self.worker.failed.connect(self.on_llm_failed)
        self.worker.start()

    def on_llm_done(self, data):
        self.overlay.hide()
        self.video_label.hide()
        self.capture_btn.hide()
        self.result_view.show()

        stretches = data.get("stretches", [])
        workout = data.get("workout", {"recommendations": []})
        recs = workout.get("recommendations", [])

        self._concatenate_text(stretches, recs)

        self.selector = workout_sel()
        self.selector.load_workout(workout)

        if recs:
            self.start_workout_btn.show()

        print("스트레칭:", stretches)
        print("운동:", recs)

    def on_llm_failed(self, message):
        self.overlay.hide()
        self.status_label.setText(message)
        print("LLM 실패:", message)

    def on_start_workout(self):
        """운동 시작 버튼 콜백 — 카메라 정리 후 WorkoutView로 전환."""
        if self.selector is None:
            return

        self.cap.release()
        self.workout_view = WorkoutView(self.selector)
        self.workout_view.show()
        self.close()

    def _concatenate_text(self, stretches, recs):
        lines = []

        lines.append("STRETCHES")
        if stretches:
            for s in stretches:
                lines.append(f"  ·  {s.get('name')}  —  {s.get('reason')}")
        else:
            lines.append("  ·  (없음)")

        lines.append("")
        lines.append("TODAY'S WORKOUT")
        if recs:
            for r in recs:
                cnt = r.get("count")
                unit = r.get("unit")
                sets = r.get("sets")
                lines.append(f"  {r.get('priority')}.  {r.get('exercise')}")
                lines.append(f"      {cnt}{unit} x {sets} sets")
                weight = r.get("weight")
                if weight:
                    lines.append(f"      권장 중량: {weight}")
                lines.append(f"      {r.get('reason')}")
                lines.append("")
        else:
            lines.append("  ·  (없음)")

        self.result_view.setText("\n".join(lines))

    def _restart(self):
        self.cap_form = CaptureForm()
        self.status_label.setText("정면 자세를 잡고 '촬영'을 누르세요")
        if not self.timer.isActive():
            self.timer.start(30)

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


class WorkoutView(QWidget):
    """
    workout_sel이 관리하는 운동 목록을 순서대로 진행하는 실행 화면.
    카메라 프레임을 ActionRecognizer(LSTM)로 운동 종류를 자동 인식하고,
    인식된 운동이 현재 목표 운동과 일치할 때만 workout_checker(get_exercise_checker)로
    rep 카운트 + 자세 피드백을 실시간으로 표시한다.
    """

    # 목표 운동과 다른 동작으로 인식된 상태가 이만큼 연속으로 유지돼야 화면에 표시
    MISMATCH_DEBOUNCE_SEC = 1.0

    def __init__(self, selector: "workout_sel"):
        super().__init__()
        self.setWindowTitle("운동 진행")
        self.setMinimumWidth(560)
        self.selector = selector
        self.checker = None
        self.recognizer = ActionRecognizer()
        # 목표 운동과 다른 동작이 감지되기 시작한 시각 (짧은 오인식 깜빡임 방지용 디바운스)
        self.mismatch_since = None

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(10)

        title = QLabel("운동 진행")
        title.setObjectName("title")
        layout.addWidget(title)

        video_card = QWidget()
        video_card.setObjectName("cardFlat")
        video_card_layout = QVBoxLayout()
        video_card_layout.setContentsMargins(14, 14, 14, 14)

        self.video_label = QLabel("카메라 준비 중...")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 440)
        video_card_layout.addWidget(self.video_label)
        video_card.setLayout(video_card_layout)
        layout.addWidget(video_card)

        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(4, 18, 4, 18)
        
        name_col = QVBoxLayout()
        self.exercise_label = QLabel("-")
        self.exercise_label.setObjectName("exerciseName")
        name_col.addWidget(self.exercise_label)
        
        self.set_label = QLabel("세트 -/-")
        self.set_label.setObjectName("caption")
        name_col.addWidget(self.set_label)

        self.recognized_label = QLabel("동작 인식 준비 중...")
        self.recognized_label.setObjectName("caption")
        name_col.addWidget(self.recognized_label)

        info_layout.addLayout(name_col)
        info_layout.addStretch()
        
        metric_col = QVBoxLayout()
        metric_col.setAlignment(Qt.AlignRight)
        self.count_label = QLabel("0")
        self.count_label.setObjectName("bigMetric")
        self.count_label.setAlignment(Qt.AlignRight)
        metric_col.addWidget(self.count_label)
        
        self.count_unit_label = QLabel("/ 10 reps")
        self.count_unit_label.setObjectName("metricUnit")
        self.count_unit_label.setAlignment(Qt.AlignRight)
        metric_col.addWidget(self.count_unit_label)
        
        info_layout.addLayout(metric_col)
        layout.addLayout(info_layout)   

        self.feedback_label = QLabel("자세를 잡고 운동을 시작하세요")
        self.feedback_label.setObjectName("feedbackGood")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setWordWrap(True)
        layout.addWidget(self.feedback_label)

        self.start_btn = QPushButton("운동 시작")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.on_start_workout)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("운동 종료")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.on_stop)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # extract_landmarks_video는 단조증가하는 타임스탬프(ms)가 필요
        self.start_time = time.perf_counter()
        self.prev_ts = -1

        # '운동 시작' 버튼을 누르기 전까지는 미리보기만 표시하고 판별은 하지 않음
        self.workout_started = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self._load_next_exercise()
        self._set_feedback("\"운동 시작\" 버튼을 눌러 시작하세요", "feedbackGood")

    def on_start_workout(self):
        self.workout_started = True
        self.start_btn.hide()
        # 대기 시간이 타임스탬프에 섞이지 않도록 시작 시점을 다시 기준으로 잡음
        self.start_time = time.perf_counter()
        self.prev_ts = -1
        self._set_feedback("자세를 잡고 운동을 시작하세요", "feedbackGood")

    def _load_next_exercise(self, work_done: bool = False):
        """workout_sel에서 다음 운동을 받아와 checker를 새로 생성."""
        item = self.selector.next_workout(work_done) if work_done else self.selector.current_workout()

        if item is None:
            self._show_all_done()
            return

        self.checker = get_exercise_checker(
            item["exercise"],
            target_reps=item.get("sets", 1),
            target_count=item.get("count", 10),
            rest_seconds=item.get("rest_seconds", 30),
        )
        self.recognizer.reset()
        self.mismatch_since = None
        self.exercise_label.setText(item["exercise"].upper())
        self.set_label.setText(f"세트 1/{item.get('sets', 1)}")
        self.count_unit_label.setText(f"/ {item.get('count', 10)} reps")
        self._set_feedback("자세를 잡고 운동을 시작하세요", "feedbackGood")
        self._set_recognized("동작 인식 준비 중...", "caption")

    def _show_all_done(self):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.video_label.setText("모든 운동을 완료했습니다!")
        self.exercise_label.setText("완료")
        self.count_label.setText("✓")
        self.count_unit_label.setText("")
        self._set_feedback("수고하셨습니다. 오늘의 운동을 모두 마쳤어요.", "feedbackGood")

    def _set_feedback(self, text, kind):
        """feedback_label의 텍스트와 상태(good/warn)를 갱신하고 QSS를 다시 적용한다.

        Qt는 objectName이 바뀌어도 스타일을 자동으로 재계산하지 않으므로
        unpolish/polish를 호출해 강제로 다시 적용해야 색이 바뀐다.
        """
        self.feedback_label.setText(text)
        self.feedback_label.setObjectName(kind)
        self.feedback_label.style().unpolish(self.feedback_label)
        self.feedback_label.style().polish(self.feedback_label)

    def _set_recognized(self, text, kind):
        """recognized_label(LSTM이 현재 인식 중인 운동)의 텍스트와 스타일을 갱신."""
        self.recognized_label.setText(text)
        self.recognized_label.setObjectName(kind)
        self.recognized_label.style().unpolish(self.recognized_label)
        self.recognized_label.style().polish(self.recognized_label)

    def _update_recognized_label(self, recognized_action):
        if recognized_action is None:
            self.mismatch_since = None
            self._set_recognized("동작 인식 준비 중...", "caption")
            return

        if recognized_action == "...":
            self.mismatch_since = None
            self._set_recognized("동작 인식 중...", "caption")
            return

        if recognized_action == self.checker.name:
            self.mismatch_since = None
            self._set_recognized(f"{recognized_action.upper()} 중...", "caption")
            return

        # 목표 운동과 다르게 인식됨 — 짧은 오인식(글리치)일 수 있으므로
        # MISMATCH_DEBOUNCE_SEC 이상 연속으로 유지될 때만 화면에 반영한다.
        now = time.time()
        if self.mismatch_since is None:
            self.mismatch_since = now
            return  # 방금 시작된 불일치, 아직은 이전 표시를 그대로 둔다

        if now - self.mismatch_since >= self.MISMATCH_DEBOUNCE_SEC:
            self._set_recognized(
                f"{recognized_action.upper()} 중... (목표 운동: {self.checker.name.upper()})",
                "feedbackWarn",
            )

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or self.checker is None:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if not self.workout_started:
            # 버튼을 누르기 전엔 판별 없이 카메라 미리보기만 표시
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(pixmap)
            return

        session = self.checker.session
        # 휴식 시간은 프레임 처리 여부와 상관없이 흘러야 하므로 먼저 갱신
        session.update_rest()

        if session.resting or session.done:
            # 휴식/완료 중엔 카메라 밖으로 나가도 되므로 포즈 인식을 건너뛴다
            self.mismatch_since = None
            self._set_recognized("쉬는 중" if session.resting else "-", "caption")
            self._refresh_status()
        else:
            ts = int((time.perf_counter() - self.start_time) * 1000)
            if ts <= self.prev_ts:
                ts = self.prev_ts + 1
            self.prev_ts = ts

            results = extract_landmarks_video(frame, ts)
            if has_landmarks(results):
                landmarks = results.pose_landmarks[0]
                frame = draw_skeleton(frame, results)
                # LSTM이 인식한 운동이 현재 목표 운동과 일치할 때만 checker가 카운트한다
                recognized_action, _, _ = update_with_recognition(
                    self.recognizer, self.checker, landmarks, w, h)
                self._update_recognized_label(recognized_action)
                self._refresh_status()
            else:
                self._set_feedback("몸 전체가 화면에 보이도록 위치를 조정하세요", "feedbackWarn")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

        if session.done:
            self._load_next_exercise(work_done=True)

    def _refresh_status(self):
        session = self.checker.session

        self.count_label.setText(str(session.count))
        self.set_label.setText(f"세트 {session.rep_count + 1}/{session.target_reps}")

        if session.resting:
            remaining = int(session.rest_remaining())
            self._set_feedback(f"휴식 중... {remaining}초 남음", "feedbackWarn")
            return

        if session.done:
            return

        errors = session.last_rep_errors
        if errors:
            self._set_feedback("  /  ".join(errors), "feedbackWarn")
        else:
            self._set_feedback("Good form!", "feedbackGood")

    def on_stop(self):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


class UserInfoForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("사용자 정보 입력")
        self.setMinimumWidth(480)
        self.result = None

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 26, 30, 26)
        layout.setSpacing(4)

        title = QLabel("맞춤 운동 추천")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("간단한 정보를 알려주시면 자세 분석 기반으로\n맞춤 스트레칭과 운동을 추천해드려요")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # ----- 나이 -----
        age_label = QLabel("나이")
        age_label.setObjectName("field")
        layout.addWidget(age_label)

        self.age_spin = QSpinBox()
        self.age_spin.setRange(10, 90)
        self.age_spin.setValue(30)
        self.age_spin.setSuffix("  세")
        layout.addWidget(self.age_spin)

        # ----- 체력 수준 -----
        level_label = QLabel("체력 수준 / 운동 경험")
        level_label.setObjectName("field")
        layout.addWidget(level_label)

        self.level_combo = QComboBox()
        self.level_combo.addItems(["초급", "중급", "고급"])
        layout.addWidget(self.level_combo)

        # ----- 운동 목표 -----
        goal_label = QLabel("운동 목표")
        goal_label.setObjectName("field")
        layout.addWidget(goal_label)

        goal_layout = QHBoxLayout()
        goal_layout.setSpacing(8)
        self.goal_group = QButtonGroup(self)

        self.goal_strength = QRadioButton("근력 강화")
        self.goal_endurance = QRadioButton("지구력")
        self.goal_strength.setChecked(True)

        for i, btn in enumerate([self.goal_strength, self.goal_endurance]):
            self.goal_group.addButton(btn, i)
            goal_layout.addWidget(btn)

        layout.addLayout(goal_layout)

        # ----- 제출 버튼 -----
        submit_btn = QPushButton("자세 촬영하러 가기")
        submit_btn.setObjectName("primary")
        submit_btn.clicked.connect(self.on_submit)
        layout.addWidget(submit_btn)

        self.setLayout(layout)

    def on_submit(self):
        goal_map = {0: "strength", 1: "endurance"}
        level_map = {"초급": "beginner", "중급": "intermediate", "고급": "advanced"}

        self.result = {
            "age": self.age_spin.value(),
            "level": level_map[self.level_combo.currentText()],
            "goal": goal_map[self.goal_group.checkedId()],
        }
        print("입력값:", self.result)

        self.camera = CameraView(self.result)
        self.camera.show()
        self.close()
