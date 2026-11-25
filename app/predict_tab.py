import numpy as np
from PySide6.QtGui import QColor, QPixmap, QIcon
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
                               QPushButton, QColorDialog, QListWidget,
                               QListWidgetItem, QLabel)

from .map_widget import MapWidget

COLORES_BASICOS = {
    "Blanco": (255, 255, 255), "Negro": (0, 0, 0),
    "Rojo": (255, 0, 0), "Verde": (0, 255, 0),
    "Azul": (0, 0, 255), "Amarillo": (255, 255, 0),
    "Magenta": (255, 0, 255), "Cian": (0, 255, 255),
}

TEXTO_INICIAL = "Elige un color para ver dónde lo coloca el SOM."


def icono_color(rgb):
    pix = QPixmap(16, 16)
    pix.fill(QColor(*rgb))
    return QIcon(pix)


class PredictTab(QWidget):
    def __init__(self, som, parent=None):
        super().__init__(parent)
        self.probados = []  # lista de (rgb 0-255, i, j)
        self.mapa = MapWidget()

        btn_elegir = QPushButton("Elegir color…")
        btn_elegir.clicked.connect(self._elegir_color)

        rejilla = QGridLayout()
        for n, (nombre, rgb) in enumerate(COLORES_BASICOS.items()):
            boton = QPushButton(nombre)
            boton.setIcon(icono_color(rgb))
            # c=rgb "congela" el color de cada botón dentro de la lambda
            boton.clicked.connect(lambda checked=False, c=rgb: self.probar_color(c))
            rejilla.addWidget(boton, n // 2, n % 2)

        self.info = QLabel(TEXTO_INICIAL)
        self.info.setWordWrap(True)
        self.lista = QListWidget()
        btn_limpiar = QPushButton("Limpiar")
        btn_limpiar.clicked.connect(self._limpiar)

        panel = QVBoxLayout()
        panel.addWidget(btn_elegir)
        panel.addWidget(QLabel("Colores básicos:"))
        panel.addLayout(rejilla)
        panel.addWidget(self.info)
        panel.addWidget(QLabel("Colores probados:"))
        panel.addWidget(self.lista)
        panel.addWidget(btn_limpiar)

        layout = QHBoxLayout(self)
        layout.addWidget(self.mapa, stretch=3)
        layout.addLayout(panel, stretch=1)

        self.set_som(som)

    def set_som(self, som):
        # Lo usaremos también para probar el modelo que entrenes tú
        self.som = som
        self.mapa.set_pesos(som.pesos)
        self._limpiar()

    def _elegir_color(self):
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self.probar_color((color.red(), color.green(), color.blue()))

    def probar_color(self, rgb):
        patron = np.array(rgb) / 255  # misma normalización que en el entrenamiento
        i, j = self.som.bmu(patron)
        distancia = np.linalg.norm(self.som.pesos[i, j] - patron)

        self.probados.append((rgb, i, j))
        self.lista.addItem(QListWidgetItem(icono_color(rgb), f"RGB{rgb} → ({i}, {j})"))
        self.info.setText(f"RGB{rgb} va a la neurona ({i}, {j}).\n"
                          f"Distancia a su peso: {distancia:.3f}")
        self._actualizar_marcadores()

    def _actualizar_marcadores(self):
        self.mapa.set_marcadores(
            [(i, j, tuple(c / 255 for c in rgb)) for rgb, i, j in self.probados])

    def _limpiar(self):
        self.probados.clear()
        self.lista.clear()
        self.info.setText(TEXTO_INICIAL)
        self._actualizar_marcadores()