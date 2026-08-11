import sys
import time

import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QComboBox, QGroupBox, QRadioButton, QButtonGroup,
    QProgressBar, QTextEdit,
)

from module.mediapipe_op import *      # CaptureForm, save_with_pose, has_landmarks, extract_landmarks_video, draw_skeleton
from module.workout_sel import *
from module.cal_angle import *
from module.ollama_op import *
from module.workout_checker import get_exercise_checker

from gui.gui_style import *


# =====================================================
# features(cal_angle 키) -> REF_RANGES 키로 변환
# (cal_angle.py 에 이미 있다면 이 함수는 지우고 import 해서 쓸 것)
# =====================================================
def features_to_metrics(front_features, side_features):
    def avg(a, b):
        vals = [v for v in (a, b) if v is not None]
        return sum(vals) / len(vals) if vals else None

    metrics = {
        "shoulder_tilt":     front_features.get("shoulder_tilt_deg"),
        "knee_valgus":       avg(front_features.get("left_knee_valgus_deg"),
                                 front_features.get("right_knee_valgus_deg")),
        "forward_head":      side_features.get("fha_deg"),
        "round_shoulder":    side_features.get("fsa_deg"),
        "thoracic_kyphosis": side_features.get("thoracic_kyphosis_deg"),
    }
    return {k: v for k, v in metrics.items() if v is not None}


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

        # 호출 1: 스트레칭 (표시 전용)
        stretches = recommend_stretches(abnormal)

        # 호출 2: 운동 처방 (workout_sel 형식)
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
        self.setStyleSheet("background-color: rgba(30, 30, 46, 255);")  # 불투명 - 카메라 가림

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("자세 분석 중...\n잠시만 기다려주세요")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #cdd6f4; font-size: 18px; font-weight: bold;")
        layout.addWidget(self.label)

        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedWidth(200)
        layout.addWidget(bar, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def resizeEvent(self, event):
        if self.parent():
            self.resize(self.parent().size())


class CameraView(QWidget):
    REST_BETWEEN_EXERCISES = 10   # 운동 간 휴식(초)

    def __init__(self, user_info: dict):
        super().__init__()
        self.setWindowTitle("자세 촬영 & 운동")
        self.user_info = user_info
        self.cap_form = CaptureForm()

        # 상태: "capture" -> "analyzing" -> "result" -> "exercise" -> "done"
        self.state = "capture"

        # 실시간 운동용
        self.selector = None
        self.checker = None
        self.between_rest = False
        self.between_rest_start = None
        self._last_rep_count = 0     # 세트 완료 감지용

        # VIDEO 타임스탬프
        self.ts_start = time.perf_counter()
        self.prev_ts = -1

        layout = QVBoxLayout()

        self.video_label = QLabel("카메라 준비 중...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        self.status_label = QLabel("정면 자세를 잡고 '촬영' 버튼을 누르세요")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)      # 여러 줄 표시 허용
        layout.addWidget(self.status_label)

        # 세트/폼 피드백 로그
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setWordWrap(True)
        self.feedback_label.hide()
        layout.addWidget(self.feedback_label)

        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setMinimumHeight(200)
        self.result_view.hide()
        layout.addWidget(self.result_view)

        # 버튼 두 개: 촬영 / 운동 시작
        btn_row = QHBoxLayout()
        self.capture_btn = QPushButton("촬영")
        self.capture_btn.clicked.connect(self.on_capture)
        btn_row.addWidget(self.capture_btn)

        self.start_btn = QPushButton("운동 시작")
        self.start_btn.clicked.connect(self.start_exercise)
        self.start_btn.hide()
        btn_row.addWidget(self.start_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_frame = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ---------- 타임스탬프 ----------
    def _next_ts(self):
        ts = int((time.perf_counter() - self.ts_start) * 1000)
        if ts <= self.prev_ts:
            ts = self.prev_ts + 1
        self.prev_ts = ts
        return ts

    # ---------- 카메라 루프 ----------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame
        h, w, _ = frame.shape

        display = frame
        if self.state == "exercise":
            display = self._run_exercise(frame, w, h)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        hh, ww, ch = rgb.shape
        qimg = QImage(rgb.data, ww, hh, ch * ww, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # ---------- 촬영 ----------
    def on_capture(self):
        if self.current_frame is None or self.state != "capture":
            return

        self.cap_form.key_released = True
        result = self.cap_form.capture_form(ord('c'), self.current_frame)

        if not self.cap_form.front_img_en:
            self.status_label.setText("정면 자세를 잡고 '촬영'을 누르세요")
        elif not self.cap_form.side_img_en:
            self.status_label.setText("측면(왼쪽) 자세를 잡고 '촬영'을 누르세요")

        if result is not None:
            front_img, side_img = result
            self.status_label.setText("촬영 완료 — 분석 중...")
            cv2.imwrite("front_raw.jpg", front_img)
            cv2.imwrite("side_raw.jpg", side_img)
            self.on_capture_done(front_img, side_img)

    def on_capture_done(self, front_img, side_img):
        self.state = "analyzing"
        self.timer.stop()
        self.video_label.clear()

        front_results = save_with_pose(front_img, "front_pose.jpg")
        side_results  = save_with_pose(side_img,  "side_pose.jpg")

        if not has_landmarks(front_results) or not has_landmarks(side_results):
            self.status_label.setText("포즈 감지 실패 — 다시 촬영")
            self._restart()
            return

        fh, fw = front_img.shape[:2]
        sh, sw = side_img.shape[:2]
        front_features = calculate_all_features(front_results.pose_landmarks[0], fw, fh)
        side_features  = calculate_all_features(side_results.pose_landmarks[0], sw, sh)
        metrics = features_to_metrics(front_features, side_features)
        print("metrics:", metrics)

        self.overlay = LoadingOverlay(self)
        self.overlay.resize(self.size())
        self.overlay.show()

        self.worker = LLMWorker(metrics, self.user_info)
        self.worker.finished.connect(self.on_llm_done)
        self.worker.failed.connect(self.on_llm_failed)
        self.worker.start()

    # ---------- LLM 결과 ----------
    def on_llm_done(self, data):
        self.overlay.hide()
        self.state = "result"
        self.capture_btn.hide()
        self.video_label.hide()
        self.result_view.show()

        stretches = data.get("stretches", [])
        workout = data.get("workout", {"recommendations": []})
        recs = sorted(workout.get("recommendations", []),
                      key=lambda x: x.get("priority", 99))

        self._concatenate_text(stretches, recs)

        # 운동 선택기에 로드
        self.selector = workout_sel()
        self.selector.load_workout(workout)
        print("스트레칭:", stretches)
        print("운동:", recs)

        # 운동 시작 버튼 노출
        self.status_label.setText("운동을 시작하려면 '운동 시작'을 누르세요")
        self.start_btn.show()

    def on_llm_failed(self, message):
        self.overlay.hide()
        self.status_label.setText(message)
        print("LLM 실패:", message)
        self._restart()

    def _concatenate_text(self, stretches, recs):
        lines = ["〈 추천 스트레칭 〉"]
        if stretches:
            for s in stretches:
                lines.append(f"· {s.get('name')} — {s.get('reason')}")
        else:
            lines.append("· (없음)")

        lines.append("")
        lines.append("〈 오늘의 운동 〉")
        if recs:
            for r in recs:
                lines.append(
                    f"{r.get('priority')}. {r.get('exercise')} "
                    f"— {r.get('count')}{r.get('unit')} × {r.get('sets')}세트 "
                    f"(휴식 {r.get('rest_seconds', 30)}초)  ({r.get('reason')})"
                )
        else:
            lines.append("· (없음)")

        self.result_view.setText("\n".join(lines))

    # ---------- 실시간 운동 ----------
    def start_exercise(self):
        if self.selector is None:
            return
        current = self.selector.current_workout()
        if current is None:
            self._all_done()
            return

        self.checker = self._make_checker(current)
        self._last_rep_count = 0
        self.state = "exercise"
        self.result_view.hide()
        self.start_btn.hide()
        self.video_label.show()
        self.feedback_label.show()
        self.feedback_label.setText("")
        self.status_label.setText(f"{current['exercise']} 시작")

        if not self.timer.isActive():
            self.timer.start(30)

    def _make_checker(self, item):
        return get_exercise_checker(
            item["exercise"],
            target_reps=item.get("sets", 1),
            target_count=item.get("count", 10),
            rest_seconds=item.get("rest_seconds", 30),
        )

    def _run_exercise(self, frame, w, h):
        # ----- 운동 간 휴식 -----
        if self.between_rest:
            remaining = self.REST_BETWEEN_EXERCISES - (time.time() - self.between_rest_start)
            if remaining <= 0:
                self.between_rest = False
                self.status_label.setText(f"{self.checker.name} 시작")
            else:
                self.status_label.setText(f"다음 운동까지 {remaining:.0f}초")
            return frame

        # ----- 실시간 판별 -----
        ts = self._next_ts()
        results = extract_landmarks_video(frame, ts)

        info = None
        if has_landmarks(results):
            landmarks = results.pose_landmarks[0]
            info = self.checker.update(landmarks, w, h)
            frame = draw_skeleton(frame, results)

        session = self.checker.session

        # ----- 세트 완료 감지 → 피드백 로그 -----
        if session.rep_count > self._last_rep_count:
            fb = "정확한 자세!" if not session.last_rep_errors \
                 else " / ".join(session.last_rep_errors)
            msg = f"[세트 {session.rep_count} 완료]  {fb}"
            print(msg)
            self.feedback_label.setText(msg)
            self._last_rep_count = session.rep_count

        # ----- 상태 표시 -----
        if session.resting:
            # 세트 간 휴식
            self.status_label.setText(
                f"{self.checker.name}  세트 {session.rep_count}/{session.target_reps} 완료\n"
                f"세트 휴식 {session.rest_remaining():.0f}초")
        else:
            # 플랭크: 남은 유지 시간 표시
            if self.checker.name == "plank" and info is not None:
                elapsed = info.get("elapsed", 0.0)
                target = info.get("target_seconds", 0)
                remaining_hold = max(0.0, target - elapsed)
                self.status_label.setText(
                    f"플랭크  세트 {session.rep_count}/{session.target_reps}  "
                    f"남은 시간 {remaining_hold:.1f}초 / {target:.0f}초")
            else:
                # 스쿼트/바이셉컬: 카운트 + 실시간 폼 경고
                warn = ""
                if session.last_rep_errors:
                    warn = "   ⚠ " + session.last_rep_errors[0]
                self.status_label.setText(
                    f"{self.checker.name}  "
                    f"세트 {session.rep_count}/{session.target_reps}  "
                    f"카운트 {session.count}/{session.target_count}{warn}")

        # ----- 한 운동 완료 → 다음 운동 -----
        if session.done:
            summary = "  ".join(f"{k}: {v}회" for k, v in session.error_counter.items())
            print(f"[{self.checker.name} 완료]  {summary}")
            self.feedback_label.setText(f"[{self.checker.name} 완료]  {summary}")

            nxt = self.selector.next_workout(work_done=True)
            if nxt is None:
                self._all_done()
            else:
                self.checker = self._make_checker(nxt)
                self._last_rep_count = 0
                self.between_rest = True
                self.between_rest_start = time.time()

        return frame

    def _all_done(self):
        self.state = "done"
        self.timer.stop()
        self.video_label.clear()
        self.feedback_label.hide()
        self.status_label.setText("모든 운동을 완료했습니다! 수고하셨습니다.")

    # ---------- 공통 ----------
    def _restart(self):
        self.cap_form = CaptureForm()
        self.state = "capture"
        self.capture_btn.show()
        self.video_label.show()
        if not self.timer.isActive():
            self.timer.start(30)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


class UserInfoForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("사용자 정보 입력")
        self.setMinimumWidth(420)
        self.result = None

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(6)

        title = QLabel("운동 추천을 위한 정보 입력")
        title.setObjectName("title")
        layout.addWidget(title)

        age_label = QLabel("나이")
        age_label.setObjectName("field")
        layout.addWidget(age_label)
        self.age_spin = QSpinBox()
        self.age_spin.setRange(10, 90)
        self.age_spin.setValue(30)
        self.age_spin.setSuffix(" 세")
        layout.addWidget(self.age_spin)

        level_label = QLabel("체력 수준 / 운동 경험")
        level_label.setObjectName("field")
        layout.addWidget(level_label)
        self.level_combo = QComboBox()
        self.level_combo.addItems(["초급", "중급", "고급"])
        layout.addWidget(self.level_combo)

        goal_box = QGroupBox("운동 목표")
        goal_layout = QHBoxLayout()
        goal_layout.setSpacing(4)
        self.goal_group = QButtonGroup(self)

        self.goal_posture   = QRadioButton("자세 교정")
        self.goal_strength  = QRadioButton("근력 강화")
        self.goal_endurance = QRadioButton("지구력")
        self.goal_posture.setChecked(True)

        for i, btn in enumerate([self.goal_posture, self.goal_strength, self.goal_endurance]):
            self.goal_group.addButton(btn, i)
            goal_layout.addWidget(btn)
        goal_box.setLayout(goal_layout)
        layout.addWidget(goal_box)

        submit_btn = QPushButton("제출")
        submit_btn.clicked.connect(self.on_submit)
        layout.addWidget(submit_btn)

        self.result_label = QLabel("정보를 입력하고 제출을 눌러주세요")
        self.result_label.setObjectName("result")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

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

        # 카메라 화면으로 전환
        self.camera = CameraView(self.result)
        self.camera.show()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if STYLE:
        app.setStyleSheet(STYLE)
    form = UserInfoForm()
    form.show()
    sys.exit(app.exec())