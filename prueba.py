import time
import numpy as np
from som import SOM

datos = np.random.default_rng(0).integers(0, 256, (100, 3)) / 255
som = SOM(lado=50, seed=42)

t = time.time()
for k in som.entrenar(datos, periodo=5000, lr0=0.4, cada=1000):
    print("iteración", k)
print(f"Tiempo: {time.time() - t:.1f}s")
print("Error cuantificación:", som.error_cuantificacion(datos))
print("Error topológico:", som.error_topologico(datos))
print("Rojo puro va a:", som.bmu(np.array([1, 0, 0])))