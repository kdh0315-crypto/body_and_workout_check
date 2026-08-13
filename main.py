"""GUI entry point for the integrated posture-analysis application."""

import sys

from PySide6.QtWidgets import QApplication

from gui.gui import UserInfoForm
from gui.gui_style import STYLE


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)

    window = UserInfoForm()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
