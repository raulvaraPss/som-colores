import numpy as np
from som import SOM

# Tu mejor configuración de la práctica
LADO = 50
PERIODO = 5000
LEARNING_RATE = 0.4
NUM_DATOS = 100
SEED = 42

rng = np.random.default_rng(SEED)
datos = rng.integers(0, 256, (NUM_DATOS, 3)) / 255  # normalizado a [0, 1]

som = SOM(lado=LADO, seed=SEED)
for k in som.entrenar(datos, PERIODO, LEARNING_RATE, cada=1000):
    print(f"Iteración {k}/{PERIODO}")

print("Error cuantificación:", som.error_cuantificacion(datos))
print("Error topológico:", som.error_topologico(datos))

som.guardar("models/pretrained.npz")
print("Guardado en models/pretrained.npz")