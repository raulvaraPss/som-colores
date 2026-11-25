from PySide6.QtWidgets import QMainWindow, QTabWidget

from .predict_tab import PredictTab


class MainWindow(QMainWindow):
    def __init__(self, som_preentrenado):
        super().__init__()
        self.setWindowTitle("SOM de colores")
        self.resize(1000, 650)

        self.tabs = QTabWidget()
        self.tabs.addTab(PredictTab(som_preentrenado), "Probar colores")
        # Aquí añadiremos la pestaña "Entrenar mi modelo"
        self.setCentralWidget(self.tabs)