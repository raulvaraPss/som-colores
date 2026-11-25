from PySide6.QtWidgets import QMainWindow, QTabWidget

from .predict_tab import PredictTab
from .train_tab import TrainTab


class MainWindow(QMainWindow):
    def __init__(self, som_preentrenado):
        super().__init__()
        self.setWindowTitle("SOM de colores")
        self.resize(1100, 700)

        self.tabs = QTabWidget()
        self.tabs.addTab(PredictTab(som_preentrenado), "Probar colores")

        self.tab_entrenar = TrainTab()
        self.tab_entrenar.probar_modelo.connect(self._probar_mi_modelo)
        self.tabs.addTab(self.tab_entrenar, "Entrenar mi modelo")

        self.tab_mi_modelo = None  # se crea la primera vez que pruebas tu modelo
        self.setCentralWidget(self.tabs)

    def _probar_mi_modelo(self, som):
        if self.tab_mi_modelo is None:
            self.tab_mi_modelo = PredictTab(som)
            self.tabs.addTab(self.tab_mi_modelo, "Probar mi modelo")
        else:
            self.tab_mi_modelo.set_som(som)
        self.tabs.setCurrentWidget(self.tab_mi_modelo)