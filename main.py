import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from som import SOM
from app.main_window import MainWindow

# Ruta relativa a este archivo, así funciona lo lances desde donde lo lances
RUTA_MODELO = Path(__file__).parent / "models" / "pretrained.npz"


def main():
    app = QApplication(sys.argv)
    ventana = MainWindow(SOM.cargar(RUTA_MODELO))
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()