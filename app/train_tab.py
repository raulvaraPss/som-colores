import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
                               QSpinBox, QDoubleSpinBox, QSlider, QPushButton,
                               QProgressBar, QLabel, QGroupBox, QFileDialog,
                               QCheckBox)

from som import SOM
from .map_widget import MapWidget

# Iteraciones que se entrenan entre fotograma y fotograma, según la velocidad
PASOS_POR_FOTOGRAMA = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Colores que el modelo no ve al entrenar, para medir los errores al final
TEST = np.random.default_rng(999).integers(0, 256, (1000, 3)) / 255

# Ayuda corta que aparece al pasar el ratón por cada campo
AYUDA = {
    "lado": "Nº de neuronas por lado (lado × lado en total).\n"
            "Más grande = más detalle y menos error de cuantificación, pero más lento.",
    "periodo": "Nº de iteraciones de entrenamiento.\n"
               "Más iteraciones = mapa más ordenado (menos error topológico), pero más lento.",
    "lr": "Cuánto se mueven los pesos hacia cada color al principio.\n"
          "Muy bajo (< 0.05) = apenas aprende. Entre 0.1 y 0.5 suele ir bien.",
    "datos": "Cuántos colores aleatorios se usan para entrenar.\n"
             "Más colores = más variedad y mejor generalización.",
    "seed": "Con semilla fija, mismos ajustes = exactamente el mismo resultado.\n"
            "Desactívala para ver cuánto varía el mapa de una vez a otra.",
}


class TrainTab(QWidget):
    # Se emite con el SOM entrenado cuando el usuario pulsa "Probar este modelo"
    probar_modelo = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.som = None
        self._gen = None  # el generador som.entrenar(...)
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fotogramas por segundo
        self._timer.timeout.connect(self._tick)

        self.mapa = MapWidget()

        # --- Hiperparámetros ---
        self.sp_lado = QSpinBox(); self.sp_lado.setRange(5, 100); self.sp_lado.setValue(30)
        self.sp_periodo = QSpinBox(); self.sp_periodo.setRange(100, 50000)
        self.sp_periodo.setSingleStep(500); self.sp_periodo.setValue(5000)
        self.sp_lr = QDoubleSpinBox(); self.sp_lr.setRange(0.001, 1.0)
        self.sp_lr.setDecimals(3); self.sp_lr.setSingleStep(0.05); self.sp_lr.setValue(0.4)
        self.sp_datos = QSpinBox(); self.sp_datos.setRange(10, 10000)
        self.sp_datos.setSingleStep(100); self.sp_datos.setValue(500)
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 99999); self.sp_seed.setValue(42)

        form = QFormLayout()

        def fila(texto, widget, ayuda):
            # Etiqueta con ⓘ; la ayuda sale al pasar el ratón por la etiqueta o por la casilla
            etiqueta = QLabel(f"{texto} ⓘ")
            etiqueta.setToolTip(ayuda)
            widget.setToolTip(ayuda)
            form.addRow(etiqueta, widget)

        fila("Lado del mapa", self.sp_lado, AYUDA["lado"])
        fila("Periodo (iteraciones)", self.sp_periodo, AYUDA["periodo"])
        fila("Learning rate inicial", self.sp_lr, AYUDA["lr"])
        fila("Nº de colores", self.sp_datos, AYUDA["datos"])

        # Semilla: casilla para activarla + número
        self.chk_seed = QCheckBox("Fija")
        self.chk_seed.setChecked(True)
        self.chk_seed.toggled.connect(self.sp_seed.setEnabled)
        self.chk_seed.setToolTip(AYUDA["seed"])
        fila_seed = QHBoxLayout()
        fila_seed.addWidget(self.chk_seed)
        fila_seed.addWidget(self.sp_seed, stretch=1)
        etiqueta_seed = QLabel("Semilla ⓘ")
        etiqueta_seed.setToolTip(AYUDA["seed"])
        self.sp_seed.setToolTip(AYUDA["seed"])
        form.addRow(etiqueta_seed, fila_seed)

        self.grupo_params = QGroupBox("Hiperparámetros")
        self.grupo_params.setLayout(form)

        # --- Velocidad (se puede cambiar mientras entrena) ---
        self.slider_vel = QSlider(Qt.Horizontal)
        self.slider_vel.setRange(0, len(PASOS_POR_FOTOGRAMA) - 1)
        self.slider_vel.setValue(4)
        self.lbl_vel = QLabel()
        self.slider_vel.valueChanged.connect(self._actualizar_lbl_vel)
        self._actualizar_lbl_vel()

        # --- Botones ---
        self.btn_entrenar = QPushButton("▶ Entrenar")
        self.btn_entrenar.clicked.connect(self._empezar)
        self.btn_pausa = QPushButton("⏸ Pausar")
        self.btn_pausa.clicked.connect(self._pausar_reanudar)
        self.btn_detener = QPushButton("⏹ Detener")
        self.btn_detener.clicked.connect(self._detener)

        # --- Estado del entrenamiento ---
        self.progreso = QProgressBar()
        self.lbl_estado = QLabel("Elige los hiperparámetros y pulsa Entrenar.")
        self.lbl_estado.setWordWrap(True)
        self.lbl_resultado = QLabel()
        self.lbl_resultado.setWordWrap(True)

        self.btn_probar = QPushButton("Probar este modelo")
        self.btn_probar.clicked.connect(lambda: self.probar_modelo.emit(self.som))
        self.btn_guardar = QPushButton("Guardar modelo…")
        self.btn_guardar.clicked.connect(self._guardar)

        botones = QHBoxLayout()
        botones.addWidget(self.btn_entrenar)
        botones.addWidget(self.btn_pausa)
        botones.addWidget(self.btn_detener)

        panel = QVBoxLayout()
        panel.addWidget(self.grupo_params)
        panel.addWidget(QLabel("Velocidad de la animación:"))
        panel.addWidget(self.slider_vel)
        panel.addWidget(self.lbl_vel)
        panel.addLayout(botones)
        panel.addWidget(self.progreso)
        panel.addWidget(self.lbl_estado)
        panel.addWidget(self.lbl_resultado)
        panel.addStretch()
        panel.addWidget(self.btn_probar)
        panel.addWidget(self.btn_guardar)

        layout = QHBoxLayout(self)
        layout.addWidget(self.mapa, stretch=3)
        layout.addLayout(panel, stretch=1)

        self._set_estado_botones(entrenando=False, hay_modelo=False)

    # ---------- control del entrenamiento ----------

    def _empezar(self):
        lado = self.sp_lado.value()
        if self.chk_seed.isChecked():
            seed = self.sp_seed.value()
        else:
            # Semilla al azar, pero la mostramos para poder repetir el resultado
            seed = int(np.random.default_rng().integers(0, 100000))
            self.sp_seed.setValue(seed)
        self.periodo = self.sp_periodo.value()
        self.lr0 = self.sp_lr.value()
        datos = np.random.default_rng(seed).integers(0, 256, (self.sp_datos.value(), 3)) / 255

        self.som = SOM(lado=lado, seed=seed)
        # cada=1: el generador se detiene en cada iteración y el temporizador
        # decide cuántas avanzar por fotograma (así la velocidad cambia en vivo)
        self._gen = self.som.entrenar(datos, self.periodo, self.lr0, cada=1)

        self.progreso.setRange(0, self.periodo)
        self.progreso.setValue(0)
        self.lbl_resultado.clear()
        self.mapa.set_pesos(self.som.pesos)  # mapa aleatorio inicial
        self._set_estado_botones(entrenando=True, hay_modelo=True)
        self._timer.start()

    def _tick(self):
        # Avanza unas cuantas iteraciones y repinta
        k = None
        try:
            for _ in range(PASOS_POR_FOTOGRAMA[self.slider_vel.value()]):
                k = next(self._gen)
        except StopIteration:
            self._terminar()
            return
        self._refrescar(k)
        if k == self.periodo:
            self._terminar()

    def _refrescar(self, k):
        self.mapa.set_pesos(self.som.pesos)
        self.progreso.setValue(k)
        eta = self.som.learning_rate(k, self.lr0, self.periodo)
        v = self.som.vecindario(k, self.som.lado / 2, self.periodo)
        self.lbl_estado.setText(f"Iteración {k} / {self.periodo}\n"
                                f"Learning rate actual: {eta:.4f}\n"
                                f"Vecindario actual: {v:.2f}")

    def _pausar_reanudar(self):
        if self._timer.isActive():
            self._timer.stop()
            self.btn_pausa.setText("▶ Reanudar")
        else:
            self._timer.start()
            self.btn_pausa.setText("⏸ Pausar")

    def _detener(self):
        self._timer.stop()
        self._gen = None
        self.lbl_estado.setText(self.lbl_estado.text() + "\n(Detenido antes de terminar)")
        self._set_estado_botones(entrenando=False, hay_modelo=True)

    def _terminar(self):
        self._timer.stop()
        self._gen = None
        self._set_estado_botones(entrenando=False, hay_modelo=True)
        self.lbl_resultado.setText("Calculando errores…")
        self.lbl_resultado.repaint()  # que se vea el texto antes del cálculo
        ec = self.som.error_cuantificacion(TEST)
        et = self.som.error_topologico(TEST)
        self.lbl_resultado.setText(f"<b>Entrenamiento terminado</b><br>"
                                   f"Error de cuantificación: {ec:.4f}<br>"
                                   f"Error topológico: {et:.3f}<br>"
                                   f"Semilla: {self.sp_seed.value()}<br>"
                                   f"<small>(medidos con 1000 colores que no ha visto)</small>")

    # ---------- utilidades ----------

    def _set_estado_botones(self, entrenando, hay_modelo):
        self.grupo_params.setEnabled(not entrenando)
        self.btn_entrenar.setEnabled(not entrenando)
        self.btn_pausa.setEnabled(entrenando)
        self.btn_pausa.setText("⏸ Pausar")
        self.btn_detener.setEnabled(entrenando)
        self.btn_probar.setEnabled(hay_modelo and not entrenando)
        self.btn_guardar.setEnabled(hay_modelo and not entrenando)

    def _actualizar_lbl_vel(self):
        n = PASOS_POR_FOTOGRAMA[self.slider_vel.value()]
        self.lbl_vel.setText(f"{n} iteraciones por fotograma")

    def _guardar(self):
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar modelo", "mi_modelo.npz",
                                              "Modelo SOM (*.npz)")
        if ruta:
            self.som.guardar(ruta)
            self.lbl_estado.setText(f"Guardado en {ruta}")