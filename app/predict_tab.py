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
        self.lista.currentRowChanged.connect(self._seleccionar)
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
        patron = np.array(rgb) / 255
        b1, b2 = self.som.dos_bmus(patron)
        distancia = np.linalg.norm(self.som.pesos[b1] - patron)
        vecinas = abs(b1[0] - b2[0]) <= 1 and abs(b1[1] - b2[1]) <= 1

        texto = (f"RGB{rgb}\n"
                 f"BMU: {b1}   (distancia {distancia:.3f})\n"
                 f"2ª BMU: {b2}\n"
                 + ("Son vecinas ✓" if vecinas else "No son vecinas ✗ (error topológico)"))

        self.probados.append({"bmu": b1, "segunda": b2,
                              "color": tuple(c / 255 for c in rgb), "texto": texto})
        item = QListWidgetItem(icono_color(rgb), f"RGB{rgb} → {b1}")
        item.setToolTip(texto)  # tooltip también al pasar el ratón por la lista
        self.lista.addItem(item)
        self.mapa.set_marcadores(self.probados)
        self.lista.setCurrentRow(len(self.probados) - 1)  # selecciona el nuevo

    def _seleccionar(self, fila):
        self.mapa.set_seleccionado(fila)
        if 0 <= fila < len(self.probados):
            self.info.setText(self.probados[fila]["texto"])

    def _limpiar(self):
        self.probados.clear()
        self.lista.clear()
        self.mapa.set_marcadores([])
        self.mapa.set_seleccionado(-1)
        self.info.setText(TEXTO_INICIAL)