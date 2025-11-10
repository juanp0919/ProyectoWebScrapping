# requerimiento4/dendogramas.py
# -*- coding: utf-8 -*-

"""
Requerimiento 4: Clustering jerárquico + dendrogramas sobre abstracts.

Funciones principales:
- generar_dendrogramas_desde_bib(...): pipeline completo desde el .bib
- generar_dendrogramas(sim_matrix, ...): si ya tienes la matriz de similitud
- recomendar_metodo(resultados): sugiere el mejor método por silhouette

Detalles:
- Preprocesa abstracts (TF-IDF palabra/bigrama, acentos→unicode).
- Similitud coseno → Distancia (1 - coseno).
- Dendrogramas para: single / complete / average linkage.
- Exporta PNG y PDF a requerimiento4/resultados/.
- Calcula número de clusters recomendado (k*) y silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import bibtexparser

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score


# --------------------------- utilidades --------------------------------- #

def _cargar_abstracts_desde_bib(bib_path: str, max_docs: Optional[int] = None) -> List[str]:
    """
    Carga y devuelve una lista de abstracts limpios desde un archivo .bib.
    Lanza FileNotFoundError si no existe o ValueError si no hay abstracts.
    """
    path = Path(bib_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f)

    abstracts: List[str] = []
    for e in db.entries:
        a = (e.get("abstract") or "").strip()
        if a:
            abstracts.append(a)

    if not abstracts:
        raise ValueError("No se encontraron abstracts en el .bib")

    if max_docs:
        abstracts = abstracts[:max_docs]

    return abstracts


def _tfidf_cosine_matrix(
    docs: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    min_df: int | float = 2,
) -> np.ndarray:
    """
    Calcula matriz de similitud coseno a partir de TF-IDF (palabra/bigrama).
    Configurada para textos en ES+EN (strip_accents='unicode'); no fija stopwords
    para no sesgar bilingüe.
    """
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,  # puede ser entero (documentos) o fracción (proporción)
    )
    X = vec.fit_transform(docs)
    return cosine_similarity(X)


def _cosine_to_condensed_dist(sim: np.ndarray) -> np.ndarray:
    """
    Convierte una matriz de similitud NxN en vector de distancia 'condensed'
    (requerido por scipy.linkage). Distancia = 1 - similitud.
    """
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)  # robustez numérica
    return squareform(D, checks=False)


# --------------------------- resultados ---------------------------------- #

@dataclass
class DendoResultado:
    method: str                 # 'single' | 'complete' | 'average'
    dendrogram_png: Path
    dendrogram_pdf: Path
    n_clusters: int             # k* que maximizó silhouette
    silhouette: float           # score silhouette para k*
    labels: np.ndarray | None = None  # etiquetas de k* (opcional)


# --------------------------- API pública --------------------------------- #

def generar_dendrogramas(
    sim_matrix: np.ndarray,
    carpeta_salida: str = "requerimiento4/resultados",
    titulo_base: str = "Dendrograma",
    methods: Tuple[str, str, str] = ("single", "complete", "average"),
    rango_k: Tuple[int, int] = (2, 20),
) -> Dict[str, DendoResultado]:
    """
    Genera dendrogramas (PNG/PDF) para los métodos indicados usando una
    matriz de similitud coseno NxN. Devuelve métricas y rutas de salida.

    Args:
        sim_matrix: Matriz de similitud coseno (NxN).
        carpeta_salida: Carpeta donde se guardan imágenes.
        titulo_base: Prefijo del título en las figuras.
        methods: Métodos de enlace a evaluar.
        rango_k: Rango [k_min, k_max] para buscar k* por silhouette.

    Returns:
        Dict[method, DendoResultado]
    """
    outdir = Path(carpeta_salida)
    outdir.mkdir(parents=True, exist_ok=True)

    condensed = _cosine_to_condensed_dist(sim_matrix)
    N = sim_matrix.shape[0]
    k_min, k_max = rango_k
    k_max = min(k_max, max(2, N))  # no más clusters que documentos

    resultados: Dict[str, DendoResultado] = {}

    for method in methods:
        # Enlace jerárquico
        Z = linkage(condensed, method=method)

        # Dendrograma
        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        dendrogram(Z, no_labels=True, color_threshold=None, ax=ax)
        ax.set_title(f"{titulo_base} — {method.title()}", fontsize=12)
        ax.set_ylabel("Distancia (1 - coseno)")
        png = outdir / f"dendogram_{method}.png"   # nombre de archivo según solicitud
        pdf = outdir / f"dendogram_{method}.pdf"
        fig.tight_layout()
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

        # Búsqueda de k* por silhouette
        best_k, best_s, best_labels = 2, -1.0, None
        if N >= 3:  # silhouette necesita al menos 2 clusters y > 2 muestras
            for k in range(k_min, k_max + 1):
                labels = fcluster(Z, k, criterion="maxclust")
                # métrica precomputed: usamos distancia = 1 - similitud
                try:
                    s = silhouette_score(1.0 - sim_matrix, labels, metric="precomputed")
                except Exception:
                    s = -1.0
                if s > best_s:
                    best_k, best_s, best_labels = k, s, labels
        else:
            best_k, best_s, best_labels = 1, 0.0, np.ones((N,), dtype=int)

        resultados[method] = DendoResultado(
            method=method,
            dendrogram_png=png,
            dendrogram_pdf=pdf,
            n_clusters=int(best_k),
            silhouette=float(best_s),
            labels=best_labels,
        )

    return resultados


def recomendar_metodo(resultados: Dict[str, DendoResultado]) -> DendoResultado:
    """
    Selecciona el método con mayor silhouette. Si empatan, devuelve el primero.
    """
    return max(resultados.values(), key=lambda r: r.silhouette)


def generar_dendrogramas_desde_bib(
    bib_path: str,
    carpeta_salida: str = "requerimiento4/resultados",
    max_docs: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    min_df: int | float = 2,
    rango_k: Tuple[int, int] = (2, 20),
) -> Dict[str, DendoResultado]:
    """
    Pipeline completo desde un archivo .bib:
    - Carga abstracts
    - TF-IDF + similitud coseno
    - Dendrogramas (single/complete/average)
    - Recomendación de k* por silhouette

    Devuelve el diccionario de resultados por método.
    """
    docs = _cargar_abstracts_desde_bib(bib_path, max_docs=max_docs)
    sim = _tfidf_cosine_matrix(
        docs, ngram_range=ngram_range, max_features=max_features, min_df=min_df
    )
    return generar_dendrogramas(
        sim_matrix=sim,
        carpeta_salida=carpeta_salida,
        titulo_base="Dendrograma (TF-IDF coseno)",
        methods=("single", "complete", "average"),
        rango_k=rango_k,
    )


# --------------------------- mini demo opcional -------------------------- #

def _demo() -> None:
    """
    Demo rápida con 6 textos sintéticos (ES/EN). No se ejecuta en import.
    Genera PNG/PDF en requerimiento4/resultados/.
    """
    docs = [
        "graph algorithms shortest path dijkstra floyd warshall",
        "shortest path in graphs using dijkstra algorithm",
        "neural networks and deep learning for images",
        "convolutional neural networks for vision",
        "procesamiento de lenguaje natural y embeddings",
        "modelos de lenguaje y representaciones semánticas",
    ]
    sim = _tfidf_cosine_matrix(docs, ngram_range=(1, 2), min_df=1)
    res = generar_dendrogramas(sim)
    best = recomendar_metodo(res)
    print("Mejor método:", best.method, "k*=", best.n_clusters, "silhouette=", round(best.silhouette, 4))


if __name__ == "__main__":
    # Descomenta para probar rápidamente con datos sintéticos:
    # _demo()
    pass
