# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# main.py
from PyQt6 import QtWidgets
from core.app import FieldWorkbenchApp


def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    fw = FieldWorkbenchApp()
    # start maximized (not fullscreen)
    try:
        QtWidgets.QApplication.processEvents()
        fw.win.showMaximized()
    except Exception:
        pass
    app.exec()


if __name__ == "__main__":
    main()
