import math

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget, QToolTip

MARGEN = 14  # píxeles alrededor del mapa, para que no se corten los marcadores


class MapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._imagen = None
        self._lado = 0
        # Cada marcador es un dict: {"bmu": (i, j), "segunda": (i, j),
        #                             "color": (r, g, b) 0-1, "texto": str}
        self._marcadores = []
        self._seleccionado = -1
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)  # recibir movimientos del ratón sin pulsar

    def set_pesos(self, pesos):
        self._lado = pesos.shape[0]
        rgb = np.ascontiguousarray((np.clip(pesos, 0, 1) * 255).astype(np.uint8))
        alto, ancho, _ = rgb.shape
        self._imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888).copy()
        self.update()

    def set_marcadores(self, marcadores):
        self._marcadores = list(marcadores)
        self.update()

    def set_seleccionado(self, indice):
        self._seleccionado = indice  # -1 = ninguno
        self.update()

    # ---------- geometría ----------

    def _area_mapa(self):
        lado_px = min(self.width(), self.height()) - 2 * MARGEN
        return QRectF((self.width() - lado_px) / 2, (self.height() - lado_px) / 2,
                      lado_px, lado_px)

    def _centro(self, area, i, j):
        celda = area.width() / self._lado
        return QPointF(area.x() + (j + 0.5) * celda, area.y() + (i + 0.5) * celda)

    def _radio(self, area):
        return max(area.width() / self._lado * 0.6, 5)

    # ---------- dibujo ----------

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._imagen is None:
            return
        area = self._area_mapa()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(area, self._imagen)
        painter.setRenderHint(QPainter.Antialiasing, True)

        radio = self._radio(area)
        for n, m in enumerate(self._marcadores):
            if n != self._seleccionado:
                self._dibujar_marcador(painter, self._centro(area, *m["bmu"]), radio, m["color"])

        # El seleccionado se dibuja el último (encima), más grande y con su 2ª BMU
        if 0 <= self._seleccionado < len(self._marcadores):
            m = self._marcadores[self._seleccionado]
            c1 = self._centro(area, *m["bmu"])
            c2 = self._centro(area, *m["segunda"])
            # Línea discontinua de la BMU a la 2ª BMU
            painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
            painter.drawLine(c1, c2)
            # 2ª BMU: anillo hueco
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(Qt.black, 3))
            painter.drawEllipse(c2, radio * 0.7, radio * 0.7)
            painter.setPen(QPen(Qt.white, 1.5, Qt.DashLine))
            painter.drawEllipse(c2, radio * 0.7, radio * 0.7)
            # BMU: más grande y con borde amarillo
            self._dibujar_marcador(painter, c1, radio * 1.3, m["color"], resaltado=True)

    def _dibujar_marcador(self, painter, centro, radio, color, resaltado=False):
        borde = QPen(QColor("#000000"), 4) if resaltado else QPen(Qt.white, 3)
        painter.setPen(borde)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centro, radio, radio)
        painter.setPen(QPen(Qt.black, 1.5))
        painter.setBrush(QColor.fromRgbF(*color))
        painter.drawEllipse(centro, radio, radio)

    # ---------- ratón ----------

    def mouseMoveEvent(self, event):
        if self._imagen is None:
            return
        area = self._area_mapa()
        radio = self._radio(area)
        pos = event.position()
        # Recorremos al revés para dar prioridad al que está dibujado encima
        for m in reversed(self._marcadores):
            c = self._centro(area, *m["bmu"])
            if math.hypot(pos.x() - c.x(), pos.y() - c.y()) <= radio:
                QToolTip.showText(event.globalPosition().toPoint(), m["texto"], self)
                return
        QToolTip.hideText()