#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para análisis de similitud entre múltiples abstracts.
Implementa comparación de textos y generación de matriz de similitud.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json

from similitud_textos import SimilitudTextos
from utils.extractor_bib import ExtratorBibTeX
from utils.exportador import Exportador

class AnalizadorSimilitud:
    """
    Clase para analizar similitud entre múltiples textos y generar matriz de similitud.
    """
    
    def __init__(self, ruta_bib: Optional[str] = None):
        """
        Inicializa el analizador de similitud.
        
        Args:
            ruta_bib: Ruta al archivo BibTeX (opcional)
        """
        self.ruta_bib = Path(ruta_bib) if ruta_bib else None
        self.similitud = SimilitudTextos()
        self._abstracts: List[Dict] = []
        
        if self.ruta_bib:
            self._cargar_abstracts()
    
    def _cargar_abstracts(self) -> None:
        """Carga los abstracts desde el archivo BibTeX."""
        if not self.ruta_bib or not self.ruta_bib.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.ruta_bib}")
            
        extractor = ExtratorBibTeX(self.ruta_bib)
        self._abstracts = extractor.get_abstracts()

        # NOTE: sanea campos vacíos y remueve abstracts vacíos de raíz
        limpios = []
        for a in self._abstracts:
            txt = (a.get("abstract") or "").strip()
            if txt:
                limpios.append({"id": str(a.get("id", "")), "abstract": txt})
        self._abstracts = limpios
        
    # ---------- Utilidades internas ----------
    def _ids_unicos(self, ids: List[str]) -> List[str]:
        """Si hay IDs duplicados, añade sufijos _2, _3, ..."""
        seen = {}
        out = []
        for i in ids:
            base = i or "doc"
            if base not in seen:
                seen[base] = 1
                out.append(base)
            else:
                seen[base] += 1
                out.append(f"{base}_{seen[base]}")
        return out

    # ---------- Implementación vectorizada cuando es posible ----------
    def _matriz_tfidf_coseno(self, textos: List[str]) -> np.ndarray:
        """Calcula matriz completa de similitud coseno TF-IDF de una sola vez."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if len(textos) == 1:
            return np.array([[1.0]])
        vec = TfidfVectorizer()
        try:
            X = vec.fit_transform([t.lower() for t in textos])
            M = cosine_similarity(X)
            return np.nan_to_num(M, nan=0.0)
        except Exception:
            # fallback: matriz identidad
            n = len(textos)
            return np.eye(n, dtype=float)

    def _matriz_semantic_lsa(self, textos: List[str]) -> np.ndarray:
        """Calcula matriz completa de similitud con LSA (semantic_embedding) de una sola vez."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.metrics.pairwise import cosine_similarity

        if len(textos) == 1:
            return np.array([[1.0]])
        try:
            vec = TfidfVectorizer(
                tokenizer=self.similitud._preprocess_text,    # usa el tokenizer robusto
                stop_words=list(self.similitud.stop_words)
            )
            X = vec.fit_transform(textos)
            if X.shape[1] < 2:
                return np.eye(len(textos), dtype=float)
            k = min(100, X.shape[1]-1)
            lsa = TruncatedSVD(n_components=k)
            Z = lsa.fit_transform(X)
            M = cosine_similarity(Z)
            return np.nan_to_num(M, nan=0.0)
        except Exception:
            n = len(textos)
            return np.eye(n, dtype=float)

    def _matriz_contextual(self, textos: List[str]) -> np.ndarray:
        """Calcula matriz con el método contextual (no vectorizable fácilmente)."""
        n = len(textos)
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            M[i, i] = 1.0
            for j in range(i+1, n):
                s = self.similitud.contextual_similarity(textos[i], textos[j])
                M[i, j] = M[j, i] = s
        return M

    def _matriz_algoritmo_par_a_par(self, textos: List[str], algoritmo: str) -> np.ndarray:
        """Construye matriz por pares para algoritmos no vectorizados (levenshtein, jaccard, ngramas)."""
        n = len(textos)
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            M[i, i] = 1.0
            for j in range(i+1, n):
                s = self.similitud.calcular_similitud(textos[i], textos[j], algoritmo)
                M[i, j] = M[j, i] = s
        return M

    def comparar_textos(self, 
                        textos: List[str], 
                        algoritmo: str = 'tfidf_coseno') -> pd.DataFrame:
        """
        Compara una lista de textos entre sí.
        
        Args:
            textos: Lista de textos a comparar
            algoritmo: Nombre del algoritmo a usar
            
        Returns:
            DataFrame con matriz de similitud
        """
        # NOTE: limpia textos None/NaN y trims
        textos = [(t or "") for t in textos]
        textos = [t.strip() for t in textos]
        textos = [t for t in textos if t]  # elimina vacíos

        if not textos:
            raise ValueError("La lista de textos está vacía después de limpieza.")

        # Selección de ruta óptima según algoritmo
        algoritmo = algoritmo.lower()
        if algoritmo == "tfidf_coseno":
            matriz = self._matriz_tfidf_coseno(textos)
        elif algoritmo in ("semantic", "semantic_embedding"):
            matriz = self._matriz_semantic_lsa(textos)
        elif algoritmo in ("contextual", "contextual_similarity"):
            matriz = self._matriz_contextual(textos)
        elif algoritmo in ("levenshtein", "jaccard", "ngramas"):
            matriz = self._matriz_algoritmo_par_a_par(textos, algoritmo)
        else:
            # fallback: usa dispatcher por pares (mantiene compatibilidad)
            matriz = self._matriz_algoritmo_par_a_par(textos, algoritmo)

        # indices genéricos si el caller no los sobreescribe
        n = len(textos)
        idx = [f"Texto {i+1}" for i in range(n)]
        return pd.DataFrame(matriz, index=idx, columns=idx)
    
    def analizar_abstracts(self, 
                           algoritmo: str = 'tfidf_coseno',
                           limite: Optional[int] = None) -> pd.DataFrame:
        """
        Analiza similitud entre abstracts del archivo BibTeX.
        
        Args:
            algoritmo: Nombre del algoritmo a usar
            limite: Número máximo de abstracts a comparar
            
        Returns:
            DataFrame con matriz de similitud
        """
        if not self._abstracts:
            raise ValueError("No hay abstracts cargados")
            
        # Limitar cantidad de abstracts si se especifica
        abstracts = self._abstracts[:limite] if limite else self._abstracts
        
        # Extraer textos y IDs
        textos = [str(a["abstract"]) for a in abstracts]
        ids = [str(a.get("id", "")) for a in abstracts]

        # Asegurar unicidad en etiquetas
        ids = self._ids_unicos(ids)

        # Calcular matriz de similitud (usa path óptimo)
        matriz = self.comparar_textos(textos, algoritmo)

        # Coloca los IDs en el índice/columnas
        matriz.index = ids
        matriz.columns = ids
        
        return matriz
    
    def exportar_resultados(self,
                            matriz: pd.DataFrame,
                            directorio: str,
                            prefijo: str = "similitud") -> Dict[str, Path]:
        """
        Exporta resultados en diferentes formatos.
        
        Args:
            matriz: DataFrame con matriz de similitud
            directorio: Directorio donde guardar archivos
            prefijo: Prefijo para nombres de archivo
            
        Returns:
            Dict con rutas a los archivos generados
        """
        dir_salida = Path(directorio)
        dir_salida.mkdir(parents=True, exist_ok=True)
        
        archivos = {}
        
        # Exportar CSV
        ruta_csv = dir_salida / f"{prefijo}_matriz.csv"
        archivos['csv'] = Exportador.a_csv(matriz, ruta_csv)
        
        # Exportar JSON (asegurando tipos serializables)
        datos_json = {
            'matriz': matriz.round(6).to_dict(orient="index"),  # NOTE: menos ruido en floats
            'metadata': {
                'dimensiones': [int(matriz.shape[0]), int(matriz.shape[1])],
                'ids': list(map(str, matriz.index))
            }
        }
        ruta_json = dir_salida / f"{prefijo}_datos.json"
        archivos['json'] = Exportador.a_json(datos_json, ruta_json)
        
        return archivos


def main():
    """Función principal para pruebas."""
    # Ruta al archivo BibTeX unificado
    ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
    
    try:
        # Crear analizador
        analizador = AnalizadorSimilitud(ruta_bib)
        
        # NOTE: `SimilitudTextos.ALGORITMOS` es de instancia, no de clase.
        algos = list(analizador.similitud.ALGORITMOS.keys())

        # Analizar con diferentes algoritmos (limitando para prueba rápida)
        for algoritmo in algos:
            print(f"\nAnalizando con {algoritmo}...")
            matriz = analizador.analizar_abstracts(algoritmo=algoritmo, limite=5)
            print("\nMatriz de similitud (shape={}):".format(matriz.shape))
            print(matriz.head())

            # Exportar resultados
            dir_salida = Path(__file__).parent / "resultados" / algoritmo
            archivos = analizador.exportar_resultados(matriz, dir_salida, algoritmo)
            print("\nArchivos generados:")
            for fmt, ruta in archivos.items():
                print(f"- {fmt}: {ruta}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
