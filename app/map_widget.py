import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget


class MapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._imagen = None
        self._lado = 0
        self._marcadores = []  # lista de (i, j, (r, g, b) entre 0 y 1)
        self.setMinimumSize(300, 300)

    def set_pesos(self, pesos):
        # Pasa los pesos (0-1) a una imagen RGB de 8 bits
        self._lado = pesos.shape[0]
        rgb = np.ascontiguousarray((np.clip(pesos, 0, 1) * 255).astype(np.uint8))
        alto, ancho, _ = rgb.shape
        self._imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888).copy()
        self.update()  # pide a Qt que repinte el widget

    def set_marcadores(self, marcadores):
        self._marcadores = list(marcadores)
        self.update()

    def _area_mapa(self):
        # El cuadrado más grande que cabe, centrado
        lado_px = min(self.width(), self.height())
        return QRectF((self.width() - lado_px) / 2, (self.height() - lado_px) / 2,
                      lado_px, lado_px)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._imagen is None:
            return
        area = self._area_mapa()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(area, self._imagen)

        # Marcadores: fila i = eje vertical, columna j = eje horizontal
        celda = area.width() / self._lado
        radio = max(celda * 0.9, 6)
        painter.setRenderHint(QPainter.Antialiasing, True)
        for i, j, color in self._marcadores:
            centro = QPointF(area.x() + (j + 0.5) * celda, area.y() + (i + 0.5) * celda)
            painter.setPen(QPen(Qt.white, 4))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(centro, radio, radio)
            painter.setPen(QPen(Qt.black, 1.5))
            painter.setBrush(QColor.fromRgbF(*color))
            painter.drawEllipse(centro, radio, radio)