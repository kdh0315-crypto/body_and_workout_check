import sys
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
from module.workout_checker import get_exercise_checker


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
        workout = prescribe_workouts(goal, level)

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
    카메라 프레임을 workout_checker(get_exercise_checker)에 넘겨
    rep 카운트 + 자세 피드백을 실시간으로 표시한다.

    LSTM(운동 자동 인식) 연결 전이라, 지금은 workout_sel이 알려주는
    운동을 순서대로 수동 진행하는 방식으로 동작한다.
    """

    def __init__(self, selector: "workout_sel"):
        super().__init__()
        self.setWindowTitle("운동 진행")
        self.setMinimumWidth(560)
        self.selector = selector
        self.checker = None

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

        self.stop_btn = QPushButton("운동 종료")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.on_stop)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self._load_next_exercise()

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
        self.exercise_label.setText(item["exercise"].upper())
        self.set_label.setText(f"세트 1/{item.get('sets', 1)}")
        self.count_unit_label.setText(f"/ {item.get('count', 10)} reps")
        self.feedback_label.setText("자세를 잡고 운동을 시작하세요")
        self.feedback_label.setObjectName("feedbackGood")
        self.feedback_label.setStyleSheet("")

    def _show_all_done(self):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.video_label.setText("모든 운동을 완료했습니다!")
        self.exercise_label.setText("완료")
        self.count_label.setText("✓")
        self.count_unit_label.setText("")
        self.feedback_label.setText("수고하셨습니다. 오늘의 운동을 모두 마쳤어요.")
        self.feedback_label.setObjectName("feedbackGood")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or self.checker is None:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb) if "pose" in globals() else None
        rgb.flags.writeable = True

        landmarks = None
        if results is not None and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

        if landmarks is not None:
            result = self.checker.update(landmarks, w, h)
            self._refresh_status(result)
        else:
            self.feedback_label.setText("몸 전체가 화면에 보이도록 위치를 조정하세요")
            self.feedback_label.setObjectName("feedbackWarn")

        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

        if self.checker.session.done:
            self._load_next_exercise(work_done=True)

    def _refresh_status(self, result):
        session = self.checker.session

        self.count_label.setText(str(session.count))
        self.set_label.setText(f"세트 {session.rep_count + 1}/{session.target_reps}")

        if session.resting:
            remaining = int(session.rest_remaining())
            self.feedback_label.setText(f"휴식 중... {remaining}초 남음")
            self.feedback_label.setObjectName("feedbackWarn")
            return

        errors = getattr(self.checker, "last_rep_errors", [])
        if errors:
            self.feedback_label.setText("  /  ".join(errors))
            self.feedback_label.setObjectName("feedbackWarn")
        else:
            self.feedback_label.setText("Good form!")
            self.feedback_label.setObjectName("feedbackGood")

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

        self.goal_posture = QRadioButton("자세 교정")
        self.goal_strength = QRadioButton("근력 강화")
        self.goal_endurance = QRadioButton("지구력")
        self.goal_posture.setChecked(True)

        for i, btn in enumerate([self.goal_posture, self.goal_strength, self.goal_endurance]):
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
        goal_map = {0: "posture", 1: "strength", 2: "endurance"}
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
