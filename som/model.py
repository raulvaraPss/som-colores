import numpy as np

class SOM:
    def __init__(self, lado, num_entradas=3, seed=None):

        # Inicialización de la red SOM
        self.lado = lado
        self.num_entradas = num_entradas
        self.rng = np.random.default_rng(seed)

        # Pesos aleatorios entre 0 y 1, igual que en tu notebook
        self.pesos = self.rng.random((lado, lado, num_entradas))

        # Coordenadas (fila, columna) de cada neurona; se calculan una sola vez
        self.ii, self.jj = np.indices((lado, lado))


# La funcion distancia calcula la distancia euclídea entre el patrón de entrada y todos los pesos de la red SOM.

    def _distancias(self, patron):

        # Distancia euclídea del patrón a las lado×lado neuronas a la vez
        return np.linalg.norm(self.pesos - patron, axis=2)

#===================================================
#==== BMU (Best Matching Unit) Neurona ganadora ====
#===================================================

    def bmu(self, patron): 

        # Calcula la distancia euclidiana entre el patrón y todos los pesos de la red
        d = self._distancias(patron)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        return int(i), int(j)                


#===================================================
#====== Funciones de aprendizaje y vecindario ======
#===================================================   

# El learning rate nos permite controlar la velocidad de aprendizaje de la red. 
    def learning_rate(self, k, lr0, periodo):
        return lr0 * (1 - k / periodo)

    
# La función de vecindario nos permite definir que neuronas se van a actualizar en cada iteración de entrenamiento.
    def vecindario(self, k, v0, periodo):
        return 1 + v0 * (1 - k / periodo)


#=================================================
#====== Paso de entrenamiento de la red SOM ======
#=================================================

    def paso(self, patron, k, lr0, periodo):
        bi, bj = self.bmu(patron)
        eta = self.learning_rate(k, lr0, periodo)
        v = self.vecindario(k, self.lado / 2, periodo)

        # Distancia en el mapa de cada neurona a la BMU
        d = np.sqrt((self.ii - bi) ** 2 + (self.jj - bj) ** 2)
        # Amortiguación gaussiana (tu función decay), 0 fuera del vecindario
        h = np.exp(-d ** 2 / (2 * v ** 2))
        h[d > v] = 0

        # Actualiza todas las neuronas a la vez
        self.pesos += eta * h[..., None] * (patron - self.pesos)


#=====================================
#====== Entrenamiento de la red ======
#=====================================
    def entrenar(self, datos, periodo, lr0, cada=20):
        for k in range(1, periodo + 1):
            patron = datos[k % len(datos)]
            self.paso(patron, k, lr0, periodo)
            if k % cada == 0 or k == periodo:
                yield k


#======================
#====== Metricas ======
#======================

    def error_cuantificacion(self, datos):
        # Media de la distancia de cada dato a su BMU
        return float(np.mean([self._distancias(x).min() for x in datos]))


    def error_topologico(self, datos):
        # Proporción de datos cuya 1ª y 2ª BMU no son vecinas
        errores = 0
        for x in datos:
            d = self._distancias(x).ravel()
            primera, segunda = np.argsort(d)[:2]
            i1, j1 = np.unravel_index(primera, (self.lado, self.lado))
            i2, j2 = np.unravel_index(segunda, (self.lado, self.lado))
            if abs(i1 - i2) > 1 or abs(j1 - j2) > 1:
                errores += 1
        return errores / len(datos)


#==============================
#====== Guardar y cargar ======
#==============================

    def guardar(self, ruta):
        np.savez(ruta, pesos=self.pesos)


    @classmethod
    def cargar(cls, ruta):
        pesos = np.load(ruta)["pesos"]
        som = cls(lado=pesos.shape[0], num_entradas=pesos.shape[2])
        som.pesos = pesos
        return som