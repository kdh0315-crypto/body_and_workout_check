import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QComboBox, QGroupBox, QRadioButton, QButtonGroup
)

from module.mediapipe_op import *      # CaptureForm, save_with_pose 등
from module.workout_sel import *

from gui.gui_style import *

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
        """촬영 완료 후 스켈레톤 추출 → 저장 → 화면 표시."""
        print("촬영 완료. user_info:", self.user_info)

        # 정면/측면 각각 스켈레톤 그려 저장 + 랜드마크 결과 반환
        front_results = save_with_pose(front_img, "front_pose.jpg")
        side_results  = save_with_pose(side_img,  "side_pose.jpg")

        # 감지 실패 처리
        front_ok = front_results.pose_landmarks is not None
        side_ok  = side_results.pose_landmarks is not None

        if not front_ok or not side_ok:
            fail = []
            if not front_ok: fail.append("정면")
            if not side_ok:  fail.append("측면")
            self.status_label.setText(
                f"{', '.join(fail)} 포즈 감지 실패 — 다시 촬영해주세요"
            )
            self._restart()
            return

        self.status_label.setText("스켈레톤 추출 완료")

        # 저장된 스켈레톤 이미지를 화면에 표시
        self._show_skeleton("front_pose.jpg", "side_pose.jpg")

        # 이후 각도 계산 → find_abnormal → ask_ollama 로 이어감
        # (다음 단계에서 front_results / side_results 를 사용)
        self.front_results = front_results
        self.side_results = side_results

    def _show_skeleton(self, front_path, side_path):
        """저장된 정면/측면 스켈레톤 이미지를 나란히 표시."""
        # 정면 이미지를 video_label 자리에 표시
        front = cv2.imread(front_path)
        rgb = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        # 촬영 버튼은 더 이상 필요 없으니 비활성화
        self.capture_btn.setEnabled(False)

    def _restart(self):
        """감지 실패 시 촬영을 처음부터 다시."""
        self.cap_form = CaptureForm()          # 촬영 상태 초기화
        self.status_label.setText("정면 자세를 잡고 '촬영'을 누르세요")
        if not self.timer.isActive():
            self.timer.start(30)               # 카메라 루프 재시작

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