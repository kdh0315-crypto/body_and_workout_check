import json
import ollama
import cv2

import time

import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QSpinBox,
    QRadioButton, QButtonGroup, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox
)

from module.ollama_op import *
from module.workout_sel import *
from module.mediapipe_op import *
from module.test_fn import *

from gui.gui import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)          # 앱 전체에 스타일 적용
    form = UserInfoForm()
    form.show()
    sys.exit(app.exec())