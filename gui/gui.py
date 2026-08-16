import sys
import time
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtWidgets import QProgressBar, QTextEdit

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QComboBox, QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QSizePolicy
)

from module.mediapipe_op import *      # CaptureForm, save_with_pose 등
from module.workout_sel import *
from module.cal_angle import *
from module.ollama_op import *
from module.workout_checker import get_exercise_checker, ActionRecognizer, update_with_recognition


from gui.gui_style import *


# 운동 선호도 조사 / 신체 측정 / 운동 자세 측정 3단계가 전부 같은 창 안에서 페이지로
# 전환되므로, 세 단계 중 가장 큰 화면(카메라 영상 640x480 + 오른쪽 레이아웃들을
# 좌우로 나란히 놓는 신체 측정/운동 진행 화면)도 잘리지 않을 만큼 넉넉하게 고정한다.
MAIN_WINDOW_SIZE = (1240, 720)


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
        # 일반 QWidget은 WA_StyledBackground 없이는 스타일시트 background-color를
        # 실제로 칠하지 않아서, 이게 없으면 뒤에 있는 위젯들이 그대로 비쳐 보인다.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgb(13, 17, 23);")

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

        # 좌우 2단: 왼쪽엔 카메라 영상, 오른쪽엔 나머지 레이아웃들(제목/안내/결과/버튼)
        layout = QHBoxLayout()
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(20)

        # 결과 화면에서는 카메라 영상 없이 오른쪽 패널만 보여줄 수 있도록 self에 보관
        self.video_card = QWidget()
        self.video_card.setObjectName("cardFlat")
        video_card_layout = QVBoxLayout()
        video_card_layout.setContentsMargins(14, 14, 14, 14)

        self.video_label = QLabel("카메라 준비 중...")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        # 최소 크기만 두면 setPixmap()마다 sizeHint가 바뀌어 레이아웃 여유 공간을
        # 조금씩 잠식하며 계속 커지는 되먹임이 생기므로, 크기 자체를 고정한다.
        self.video_label.setFixedSize(640, 480)
        video_card_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.video_card.setLayout(video_card_layout)
        layout.addWidget(self.video_card)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        title = QLabel("자세 촬영")
        title.setObjectName("title")
        right_layout.addWidget(title)

        self.status_label = QLabel("정면 자세를 잡고 '촬영' 버튼을 누르세요")
        self.status_label.setObjectName("subtitle")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        self.result_view = QTextEdit()
        self.result_view.setObjectName("resultView")
        self.result_view.setReadOnly(True)
        self.result_view.setMinimumHeight(200)
        # 결과 화면에서 남는 세로 공간을 이 위젯이 채우도록 명시 (아래쪽에 빈 여백이 안 남게)
        self.result_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_view.hide()
        right_layout.addWidget(self.result_view)

        self.capture_btn = QPushButton("촬영하기")
        self.capture_btn.setObjectName("primary")
        self.capture_btn.clicked.connect(self.on_capture)
        right_layout.addWidget(self.capture_btn)

        self.start_workout_btn = QPushButton("운동 시작하기")
        self.start_workout_btn.setObjectName("primary")
        self.start_workout_btn.clicked.connect(self.on_start_workout)
        self.start_workout_btn.hide()
        right_layout.addWidget(self.start_workout_btn)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        # 카메라 영상 카드가 숨겨졌을 때(결과 화면) 오른쪽 패널이 빈 공간을 채우도록 확장 허용
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(right_panel)

        # 촬영/결과 화면을 별도 페이지로 감싸서, 운동 시작 시 창을 새로 띄우는 대신
        # 같은 (고정 크기) 창 안에서 WorkoutView 페이지로 전환할 수 있게 한다.
        capture_page = QWidget()
        capture_page.setLayout(layout)

        self.stack = QStackedWidget()
        self.stack.addWidget(capture_page)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.stack)
        self.setLayout(outer_layout)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_frame = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.selector = None
        self.workout_view = None

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
        self.overlay.raise_()  # 다른 위젯 뒤로 그려지는 경우를 막기 위해 확실히 맨 위로

        self.worker = LLMWorker(metrics, self.user_info)
        self.worker.finished.connect(self.on_llm_done)
        self.worker.failed.connect(self.on_llm_failed)
        self.worker.start()

    def on_llm_done(self, data):
        self.overlay.hide()
        # 카메라 영상은 더 안 쓰니 카드 전체를 접어서 오른쪽 결과 패널이 폭을 다 쓰게 한다.
        self.video_card.hide()
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
        """운동 시작 버튼 콜백 — 카메라 정리 후 같은 창 안에서 WorkoutView 페이지로 전환."""
        if self.selector is None:
            return

        self.cap.release()
        self.window().setWindowTitle("운동 진행")
        self.workout_view = WorkoutView(self.selector)
        self.stack.addWidget(self.workout_view)
        self.stack.setCurrentWidget(self.workout_view)

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

    def release_resources(self):
        """카메라 정리 + (있다면) 내장된 WorkoutView 페이지까지 정리.
        이 위젯도 상위 페이지 안에 내장될 수 있어 상위의 closeEvent에서 재사용한다."""
        if self.cap.isOpened():
            self.cap.release()
        if self.workout_view is not None:
            self.workout_view.release_resources()

    def closeEvent(self, event):
        self.release_resources()
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

    # 운동 종목이 바뀔 때(예: 스쿼트 -> 런지) 주는 고정 휴식 시간
    BETWEEN_EXERCISE_REST_SEC = 60

    def __init__(self, selector: "workout_sel"):
        super().__init__()
        self.setWindowTitle("운동 진행")
        self.setMinimumWidth(560)
        self.selector = selector
        self.checker = None
        self.recognizer = ActionRecognizer()
        # 목표 운동과 다른 동작이 감지되기 시작한 시각 (짧은 오인식 깜빡임 방지용 디바운스)
        self.mismatch_since = None
        # None이 아니면 "종목 전환 휴식" 중 — 값은 휴식이 시작된 시각
        self.between_rest_start = None

        # 좌우 2단: 왼쪽엔 카메라 영상, 오른쪽엔 나머지 레이아웃들(제목/상태/피드백/버튼)
        layout = QHBoxLayout()
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(20)

        video_card = QWidget()
        video_card.setObjectName("cardFlat")
        video_card_layout = QVBoxLayout()
        video_card_layout.setContentsMargins(14, 14, 14, 14)

        self.video_label = QLabel("카메라 준비 중...")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        # 최소 크기만 두면 setPixmap()마다 sizeHint가 바뀌어 레이아웃 여유 공간을
        # 조금씩 잠식하며 계속 커지는 되먹임이 생기므로, 크기 자체를 고정한다.
        self.video_label.setFixedSize(640, 440)
        video_card_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        video_card.setLayout(video_card_layout)
        layout.addWidget(video_card)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        title = QLabel("운동 진행")
        title.setObjectName("title")
        right_layout.addWidget(title)

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
        right_layout.addLayout(info_layout)

        self.feedback_label = QLabel("자세를 잡고 운동을 시작하세요")
        self.feedback_label.setObjectName("feedbackGood")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setWordWrap(True)
        right_layout.addWidget(self.feedback_label)

        self.start_btn = QPushButton("운동 시작")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.on_start_workout)
        right_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("운동 종료")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.on_stop)
        right_layout.addWidget(self.stop_btn)

        right_layout.addStretch()

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel)

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

    def _has_next_exercise(self):
        """다음 운동이 남아있는지 미리 확인 (workout_sel 상태를 옮기지 않고 들여다보기만 함)."""
        return (self.selector.state == 'work'
                and self.selector.work_cnt + 1 < len(self.selector.workouts))

    def _start_between_exercise_rest(self):
        """운동 종목이 바뀌기 전 고정 휴식을 시작한다."""
        self.between_rest_start = time.time()
        self._set_recognized("-", "caption")

    def _load_next_exercise(self, work_done: bool = False):
        """workout_sel에서 다음 운동을 받아와 checker를 새로 생성."""
        self.between_rest_start = None
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

    @staticmethod
    def _format_set_summary(error_counter):
        """방금 끝난 세트의 에러 집계(dict)를 한 줄 요약 텍스트로."""
        if not error_counter:
            return "이번 세트 기록 없음"
        parts = [f"{name} {count}회" for name, count in error_counter.items()]
        return "이번 세트: " + "  ·  ".join(parts)

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

        if self.between_rest_start is not None:
            # 종목 전환 휴식 중엔 이전 checker를 건드리지 않고 카운트다운만 표시
            remaining = self.BETWEEN_EXERCISE_REST_SEC - (time.time() - self.between_rest_start)
            if remaining <= 0:
                self._load_next_exercise(work_done=True)
            else:
                summary = self._format_set_summary(self.checker.session.last_set_error_counter)
                self._set_feedback(f"{summary}\n다음 운동까지 {int(remaining)}초 남음", "feedbackWarn")

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
            # 다음 종목이 남아있으면 곧장 넘어가지 않고 고정 휴식을 먼저 준다
            if self._has_next_exercise():
                self._start_between_exercise_rest()
            else:
                self._load_next_exercise(work_done=True)

    def _refresh_status(self):
        session = self.checker.session

        self.count_label.setText(str(session.count))
        self.set_label.setText(f"세트 {session.rep_count + 1}/{session.target_reps}")

        if session.resting:
            remaining = int(session.rest_remaining())
            summary = self._format_set_summary(session.last_set_error_counter)
            self._set_feedback(f"{summary}\n휴식 중... {remaining}초 남음", "feedbackWarn")
            return

        if session.done:
            summary = self._format_set_summary(session.last_set_error_counter)
            self._set_feedback(summary, "feedbackGood")
            return

        errors = session.last_rep_errors
        if errors:
            self._set_feedback("  /  ".join(errors), "feedbackWarn")
        else:
            self._set_feedback("Good form!", "feedbackGood")

    def on_stop(self):
        self.release_resources()
        # WorkoutView는 독립된 창이 아니라 상위 페이지 안에 내장된 페이지라서,
        # self.close()는 페이지만 숨길 뿐 창을 안 닫는다. 최상위 창을 직접 닫아야 한다.
        self.window().close()

    def release_resources(self):
        """타이머/카메라 정리. 이 위젯이 내장된 페이지라 자체 closeEvent가 안 불릴 수
        있으므로, 상위 페이지의 closeEvent에서도 직접 호출할 수 있도록 메서드로 분리."""
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()

    def closeEvent(self, event):
        self.release_resources()
        event.accept()


class UserInfoForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("사용자 정보 입력")
        # 3단계(운동 선호도 조사/신체 측정/운동 자세 측정) 전체를 같은 창에서 페이지로
        # 전환하므로, 가장 큰 페이지 기준으로 창 크기를 고정해 어느 단계에서도 안 잘리게 한다.
        self.setFixedSize(*MAIN_WINDOW_SIZE)
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

        # 입력 폼도 별도 페이지로 감싸서, 제출 시 창을 새로 띄우는 대신
        # 같은 (고정 크기) 창 안에서 CameraView 페이지로 전환할 수 있게 한다.
        form_page = QWidget()
        form_page.setLayout(layout)

        self.stack = QStackedWidget()
        self.stack.addWidget(form_page)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.stack)
        self.setLayout(outer_layout)

        self.camera = None

    def on_submit(self):
        goal_map = {0: "strength", 1: "endurance"}
        level_map = {"초급": "beginner", "중급": "intermediate", "고급": "advanced"}

        self.result = {
            "age": self.age_spin.value(),
            "level": level_map[self.level_combo.currentText()],
            "goal": goal_map[self.goal_group.checkedId()],
        }
        print("입력값:", self.result)

        self.setWindowTitle("자세 촬영")
        self.camera = CameraView(self.result)
        self.stack.addWidget(self.camera)
        self.stack.setCurrentWidget(self.camera)

    def closeEvent(self, event):
        if self.camera is not None:
            self.camera.release_resources()
        event.accept()
