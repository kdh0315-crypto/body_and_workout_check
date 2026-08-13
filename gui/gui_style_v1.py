import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QSpinBox,
    QRadioButton, QButtonGroup, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox
)
from PySide6.QtCore import Qt

STYLE = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    font-size: 14px;
}
QLabel#title {
    font-size: 22px;
    font-weight: bold;
    color: #89b4fa;
    padding-bottom: 4px;
}
QLabel#field {
    color: #a6adc8;
    font-size: 13px;
    padding-top: 6px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px;
    font-size: 13px;
    color: #a6adc8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
QComboBox, QSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px 10px;
    min-height: 20px;
}
QComboBox:hover, QSpinBox:hover {
    border: 1px solid #89b4fa;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    outline: none;
}
QRadioButton { padding: 6px; spacing: 8px; }
QRadioButton::indicator {
    width: 16px; height: 16px; border-radius: 9px;
    border: 2px solid #45475a;
}
QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border: 2px solid #89b4fa;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-size: 15px;
    font-weight: bold;
    margin-top: 8px;
}
QPushButton:hover { background-color: #a6c8ff; }
QPushButton:pressed { background-color: #74a0e8; }
QLabel#result {
    background-color: #313244;
    border-radius: 8px;
    padding: 12px;
    color: #a6e3a1;
    margin-top: 8px;
}
"""