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
        self.setStyleSheet("background-color: rgba(30, 30, 46, 200);")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("자세 분석 중...\n잠시만 기다려주세요")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #cdd6f4; font-size: 18px; font-weight: bold;")
        layout.addWidget(self.label)

        # 진행 표시 바 (무한 로딩 애니메이션)
        bar = QProgressBar()
        bar.setRange(0, 0)          # 0,0이면 값 없는 무한 애니메이션
        bar.setFixedWidth(200)
        layout.addWidget(bar, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def resizeEvent(self, event):
        # 부모 크기에 꽉 차게
        self.resize(self.parent().size())

class CameraView(QWidget):
    def __init__(self, user_info: dict):
        super().__init__()
        self.setWindowTitle("자세 촬영")
        self.user_info = user_info          # 폼에서 넘어온 age/level/goal
        self.cap_form = CaptureForm()       # 앞서 만든 촬영 상태 관리 클래스
        

        layout = QVBoxLayout()

        # 영상이 표시될 라벨
        self.video_label = QLabel("카메라 준비 중...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        # 안내 문구
        self.status_label = QLabel("정면 자세를 잡고 '촬영' 버튼을 누르세요")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setMinimumHeight(200)
        self.result_view.hide()          # 결과 나오기 전엔 숨김
        layout.addWidget(self.result_view)

        # 촬영 버튼 (키보드 c 대신 버튼으로)
        self.capture_btn = QPushButton("촬영")
        self.capture_btn.clicked.connect(self.on_capture)
        layout.addWidget(self.capture_btn)

        self.setLayout(layout)

        # 카메라 열기
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 현재 프레임을 담아둘 변수 (촬영 버튼이 참조)
        self.current_frame = None

        # 타이머: 30ms마다 update_frame 실행 (약 33fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        """타이머가 주기적으로 부르는 함수 — 프레임 한 장을 읽어 화면에 표시."""
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)          # 거울 모드
        self.current_frame = frame          # 촬영 버튼이 쓸 수 있게 저장

        # OpenCV(BGR) → Qt(RGB) 변환 후 QLabel에 표시
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def on_capture(self):
        """촬영 버튼 콜백 — 현재 프레임을 CaptureForm에 넘김."""
        if self.current_frame is None:
            return

        # 버튼 방식이라 key 대신 항상 촬영 신호를 준다
        # CaptureForm이 key==ord('c')를 기대하므로 그 값을 넣어줌
        self.cap_form.key_released = True
        result = self.cap_form.capture_form(ord('c'), self.current_frame)

        # 안내 문구 갱신
        if not self.cap_form.front_img_en:
            self.status_label.setText("정면 자세를 잡고 '촬영'을 누르세요")
        elif not self.cap_form.side_img_en:
            self.status_label.setText("측면 자세를 잡고 '촬영'을 누르세요")

        if result is not None:
            front_img, side_img = result
            self.status_label.setText("촬영 완료 — 분석 중...")
            cv2.imwrite("front_raw.jpg", front_img)
            cv2.imwrite("side_raw.jpg", side_img)
            self.timer.stop()               # 촬영 끝나면 타이머 정지
            self.on_capture_done(front_img, side_img)

    def on_capture_done(self, front_img, side_img):
        """촬영 완료 → 카메라 정지 → 각도 계산 → 로딩 표시 → LLM 백그라운드 실행."""
        print("촬영 완료. user_info:", self.user_info)

        # 카메라 루프 정지 + 화면 비우기 (로딩 중 영상 안 보이게)
        self.timer.stop()
        self.video_label.clear()

        # 1) 스켈레톤 저장 + 랜드마크 추출
        front_results = save_with_pose(front_img, "front_pose.jpg")
        side_results  = save_with_pose(side_img,  "side_pose.jpg")

        if not has_landmarks(front_results) or not has_landmarks(side_results):
            self.status_label.setText("포즈 감지 실패 — 다시 촬영")
            self._restart()
            return

        # 2) 각도 계산 → metrics 변환
        fh, fw = front_img.shape[:2]
        sh, sw = side_img.shape[:2]
        front_features = calculate_all_features(front_results.pose_landmarks[0], fw, fh)
        side_features  = calculate_all_features(side_results.pose_landmarks[0], sw, sh)


        print( "FHA Rule:", side_features.get("fha_deg"), "/", classify_fha(side_features.get("fha_deg")) )
        print("front_features:", front_features)
        print("side_features:", side_features)

        metrics = features_to_metrics(front_features, side_features)
        
        print("metrics:", metrics)

        # 3) 로딩 오버레이 표시
        self.overlay = LoadingOverlay(self)
        self.overlay.resize(self.size())
        self.overlay.show()

        # 4) LLM을 백그라운드 스레드에서 실행
        self.worker = LLMWorker(metrics, self.user_info)   # user_info 추가
        self.worker.finished.connect(self.on_llm_done)
        self.worker.failed.connect(self.on_llm_failed)
        self.worker.start()

    def on_llm_done(self, data):
        """LLM 성공 — 로딩 닫고 스트레칭 + 운동 결과 표시."""
        self.overlay.hide()
        self.video_label.hide()          # 카메라 자리 숨김
        self.capture_btn.hide()          # 촬영 버튼 숨김
        self.result_view.show()          # 결과 영역 표시

        stretches = data.get("stretches", [])
        workout = data.get("workout", {"recommendations": []})
        recs = workout.get("recommendations", [])

        # --- 결과 텍스트 조립 ---
        self._concatenate_text(stretches, recs)

        # 운동 선택기에 로드 (이후 운동 진행에 사용)
        self.selector = workout_sel()
        self.selector.load_workout(workout)
        print("스트레칭:", stretches)
        print("운동:", recs)

    def on_llm_failed(self, message):
        """LLM 실패 — 로딩 닫고 메시지 표시."""
        self.overlay.hide()
        self.status_label.setText(message)
        print("LLM 실패:", message)

    def _concatenate_text(self, stretches, recs):
        lines = []

        lines.append("〈 추천 스트레칭 〉")
        if stretches:
            for s in stretches:
                lines.append(f"· {s.get('name')} — {s.get('reason')}")
        else:
            lines.append("· (없음)")

        lines.append("")
        lines.append("〈 오늘의 운동 〉")
        if recs:
            for r in recs:
                cnt = r.get("count")
                unit = r.get("unit")
                sets = r.get("sets")
                lines.append(
                    f"{r.get('priority')}. {r.get('exercise')} "
                    f"— {cnt}{unit} × {sets}세트  ({r.get('reason')})"
                )
        else:
            lines.append("· (없음)")

        self.result_view.setText("\n".join(lines))


    def _show_result(self, image_path):
        """저장된 스켈레톤 이미지를 화면에 표시."""
        img = cv2.imread(image_path)
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.capture_btn.setEnabled(False)   # 촬영 끝났으니 버튼 비활성화

    def _restart(self):
        """감지 실패 시 촬영 처음부터."""
        self.cap_form = CaptureForm()
        self.status_label.setText("정면 자세를 잡고 '촬영'을 누르세요")
        if not self.timer.isActive():
            self.timer.start(30)

    def closeEvent(self, event):
        """창이 닫힐 때 카메라 해제."""
        self.cap.release()
        event.accept()

class UserInfoForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("사용자 정보 입력")
        self.setMinimumWidth(420)
        self.result = None

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 24)   # 창 안쪽 여백 - 왼, 위, 오, 아래 순
        layout.setSpacing(6)                          # 위젯 사이 간격

        # 제목
        title = QLabel("운동 추천을 위한 정보 입력")
        title.setObjectName("title")                  # QSS에서 #title로 지정한 것과 연결
        layout.addWidget(title)

        # 나이
        age_label = QLabel("나이")
        age_label.setObjectName("field")
        layout.addWidget(age_label)
        self.age_spin = QSpinBox()
        self.age_spin.setRange(10, 90)
        self.age_spin.setValue(30)
        self.age_spin.setSuffix(" 세")
        layout.addWidget(self.age_spin)

        # 체력 수준
        level_label = QLabel("체력 수준 / 운동 경험")
        level_label.setObjectName("field")
        layout.addWidget(level_label)
        self.level_combo = QComboBox()
        self.level_combo.addItems(["초급", "중급", "고급"])
        layout.addWidget(self.level_combo)

        # 목표
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

        # 제출 버튼
        submit_btn = QPushButton("제출")
        submit_btn.clicked.connect(self.on_submit)
        layout.addWidget(submit_btn)

        # 결과 표시
        self.result_label = QLabel("정보를 입력하고 제출을 눌러주세요")
        self.result_label.setObjectName("result")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def on_submit(self):
        goal_map = {0: "posture", 1: "strength", 2: "endurance"}
        level_map = {"초급": "beginner", "중급": "intermediate", "고급": "advanced"}
        goal_kr = ["자세 교정", "근력 강화", "지구력"]

        self.result = {
            "age": self.age_spin.value(),
            "level": level_map[self.level_combo.currentText()],
            "goal": goal_map[self.goal_group.checkedId()],
        }
        print("입력값:", self.result)
        self.result_label.setText(
            f"나이 {self.result['age']}세  ·  "
            f"{self.level_combo.currentText()}  ·  "
            f"{goal_kr[self.goal_group.checkedId()]}"
        )

        # 카메라 화면으로 전환
        self.camera = CameraView(self.result)   # 입력값을 넘김
        self.camera.show()
        self.close()                             # 입력 폼 닫기