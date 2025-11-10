#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script de prueba para los algoritmos de similitud."""

import time
import itertools
import numpy as np
import pandas as pd

from similitud_textos import SimilitudTextos

# Parámetros de prueba
N_GRAMAS = 3  # n para el algoritmo 'ngramas'


def matriz_por_pares(textos, fn):
    """Construye matriz simétrica de similitud usando una función binaria fn(t1, t2)."""
    n = len(textos)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = fn(textos[i], textos[j])
    return M


def probar_similitud():
    """Prueba todos los algoritmos de similitud con textos de ejemplo."""
    # Textos de ejemplo con más contexto
    textos = [
        """Los algoritmos de machine learning son fundamentales en el análisis de datos.
        Las técnicas modernas permiten procesar grandes volúmenes de información y extraer patrones.""",
        """El aprendizaje automático es esencial para analizar grandes cantidades de datos.
        Los modelos aprenden patrones y pueden hacer predicciones basadas en información histórica.""",
        """La física cuántica estudia el comportamiento de partículas subatómicas.
        Esta rama de la física describe fenómenos a escala microscópica."""
    ]

    ids = [f"Texto {i+1}" for i in range(len(textos))]
    sim = SimilitudTextos()

    resumen_rows = []

    # Iterar algoritmos disponibles
    for nombre in sim.ALGORITMOS.keys():
        print(f"\n===== Algoritmo: {nombre} =====")

        # Adaptador por si necesitamos parámetros específicos
        if nombre == "ngramas":
            fn = lambda a, b: sim.ngramas(a, b, n=N_GRAMAS)
        else:
            fn = lambda a, b, _nombre=nombre: sim.calcular_similitud(a, b, _nombre)

        # Medición de tiempo
        t0 = time.perf_counter()
        M = matriz_por_pares(textos, fn)
        dt = time.perf_counter() - t0

        # Mostrar matriz
        df = pd.DataFrame(M, index=ids, columns=ids)
        print(df.round(4))

        # Resumen estadístico (sin diagonal)
        valores = [M[i, j] for i, j in itertools.product(range(len(textos)), repeat=2) if i != j]
        mean_sim = float(np.mean(valores)) if valores else 0.0
        std_sim = float(np.std(valores)) if valores else 0.0

        print(f"\nResumen {nombre}:")
        print(f"- Promedio similitud (off-diagonal): {mean_sim:.4f}")
        print(f"- Desviación estándar:              {std_sim:.4f}")
        print(f"- Tiempo de ejecución:              {dt*1000:.2f} ms")

        resumen_rows.append({
            "algoritmo": nombre,
            "promedio_offdiag": mean_sim,
            "std_offdiag": std_sim,
            "tiempo_ms": dt * 1000.0
        })

    # Tabla comparativa final
    resumen_df = pd.DataFrame(resumen_rows).sort_values(by=["promedio_offdiag"], ascending=False)
    print("\n===== Comparativa global =====")
    print(resumen_df.to_string(index=False, formatters={
        "promedio_offdiag": "{:.4f}".format,
        "std_offdiag": "{:.4f}".format,
        "tiempo_ms": "{:.2f}".format
    }))


if __name__ == "__main__":
    probar_similitud()
