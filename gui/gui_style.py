STYLE = """
/* =========================================================
   컬러 팔레트
   배경   : #0d1117 (거의 검정에 가까운 남색) / #141a24 (카드)
   강조   : #34d399 (에메랄드 그린 - 활력/건강)
   보조강조: #60a5fa (블루 - 정보성 요소)
   경고   : #fbbf24 (앰버 - 피드백 경고)
   텍스트 : #e6edf3 (밝은 회백) / #8b949e (보조 텍스트)
   ========================================================= */

QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', 'Pretendard', 'Malgun Gothic', sans-serif;
    font-size: 14px;
}

/* ================= 타이포그래피 ================= */
QLabel#title {
    font-size: 26px;
    font-weight: 800;
    color: #f0f6fc;
    letter-spacing: -0.5px;
}
QLabel#subtitle {
    font-size: 13px;
    color: #8b949e;
    padding-bottom: 4px;
}
QLabel#field {
    color: #34d399;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding-top: 22px;
    padding-bottom: 4px;
}
QLabel#status {
    font-size: 14px;
    color: #8b949e;
}
QLabel#caption {
    font-size: 12px;
    color: #6e7681;
}
QLabel#bigMetric {
    font-size: 56px;
    font-weight: 800;
    color: #34d399;
}
QLabel#metricUnit {
    font-size: 15px;
    color: #8b949e;
}
QLabel#exerciseName {
    font-size: 22px;
    font-weight: 800;
    color: #f0f6fc;
}
QLabel#feedbackGood {
    font-size: 15px;
    font-weight: 700;
    color: #34d399;
    background-color: rgba(52, 211, 153, 0.12);
    border-radius: 10px;
    padding: 10px 14px;
}
QLabel#feedbackWarn {
    font-size: 15px;
    font-weight: 700;
    color: #fbbf24;
    background-color: rgba(251, 191, 36, 0.12);
    border-radius: 10px;
    padding: 10px 14px;
}

/* ================= 카드 패널 (정보입력 페이지는 카드 미사용) ================= */
QWidget#card {
    background-color: transparent;
    border: none;
}
QWidget#cardFlat {
    background-color: #141a24;
    border-radius: 14px;
}
QWidget#heroCard {
    background-color: transparent;
    border: none;
}

/* ================= 그룹박스 ================= */
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 14px;
    margin-top: 16px;
    padding: 16px 14px 14px 14px;
    font-size: 13px;
    font-weight: 700;
    color: #8b949e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #34d399;
}

/* ================= 입력 위젯 ================= */
QComboBox, QSpinBox {
    background-color: #161b22;
    border: 1.5px solid #21262d;
    border-radius: 12px;
    padding: 11px 14px;
    min-height: 20px;
    color: #e6edf3;
    selection-background-color: #34d399;
}
QComboBox:hover, QSpinBox:hover {
    border: 1.5px solid #34d399;
}
QComboBox:focus, QSpinBox:focus {
    border: 1.5px solid #34d399;
    background-color: #1c222b;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    selection-background-color: #34d399;
    selection-color: #0d1117;
    outline: none;
    padding: 6px;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 20px;
    border: none;
    background: transparent;
}

/* ================= 라디오 버튼: 선택 칩 ================= */
QRadioButton {
    padding: 12px 16px;
    spacing: 8px;
    background-color: #161b22;
    border: 1.5px solid #21262d;
    border-radius: 12px;
    font-weight: 600;
}
QRadioButton:hover {
    border: 1.5px solid #34d399;
}
QRadioButton:checked {
    background-color: rgba(52, 211, 153, 0.10);
    border: 1.5px solid #34d399;
    color: #34d399;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 9px;
    border: 2px solid #6e7681;
}
QRadioButton::indicator:checked {
    background-color: #34d399;
    border: 2px solid #34d399;
}

/* ================= 버튼: 주 액션 ================= */
QPushButton#primary {
    background-color: #34d399;
    color: #0d1117;
    border: none;
    border-radius: 12px;
    padding: 15px;
    font-size: 15px;
    font-weight: 800;
    margin-top: 12px;
}
QPushButton#primary:hover   { background-color: #4ee0ac; }
QPushButton#primary:pressed { background-color: #22b381; }
QPushButton#primary:disabled {
    background-color: #21262d;
    color: #6e7681;
}

/* ================= 버튼: 보조 액션 ================= */
QPushButton#secondary {
    background-color: #21262d;
    color: #e6edf3;
    border: none;
    border-radius: 12px;
    padding: 13px;
    font-size: 14px;
    font-weight: 700;
    margin-top: 8px;
}
QPushButton#secondary:hover   { background-color: #2d333b; }
QPushButton#secondary:pressed { background-color: #161b22; }

/* ================= 버튼: 위험/종료 액션 ================= */
QPushButton#danger {
    background-color: transparent;
    color: #f85149;
    border: 1.5px solid #3d2224;
    border-radius: 12px;
    padding: 11px;
    font-size: 13px;
    font-weight: 700;
}
QPushButton#danger:hover { background-color: rgba(248, 81, 73, 0.10); }

/* 기본 버튼은 primary와 동일 */
QPushButton {
    background-color: #34d399;
    color: #0d1117;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 15px;
    font-weight: 800;
    margin-top: 10px;
}
QPushButton:hover   { background-color: #4ee0ac; }
QPushButton:pressed { background-color: #22b381; }

/* ================= 비디오 프리뷰 ================= */
QLabel#video {
    background-color: #05070a;
    border: 2px solid #21262d;
    border-radius: 16px;
    color: #6e7681;
}

/* ================= 결과 표시 ================= */
QLabel#result {
    background-color: #161b22;
    border-radius: 12px;
    padding: 16px;
    color: #34d399;
    margin-top: 10px;
    font-weight: 700;
}
QTextEdit#resultView {
    background-color: #141a24;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 16px;
    font-size: 14px;
    line-height: 160%;
    color: #e6edf3;
}

/* ================= 진행 바 ================= */
QProgressBar {
    border: none;
    border-radius: 5px;
    background-color: #21262d;
    max-height: 10px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #34d399;
    border-radius: 5px;
}

QProgressBar#setProgress {
    max-height: 14px;
    border-radius: 7px;
}
QProgressBar#setProgress::chunk {
    border-radius: 7px;
}
"""
