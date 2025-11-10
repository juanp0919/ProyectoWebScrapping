#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimilitudTextos: Implementación de algoritmos de similitud de textos.
Incluye Levenshtein, Jaccard, TF-IDF Coseno, N-gramas, Análisis Semántico (LSA) y Contextual (ventanas).
"""

from typing import Dict, List, Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import Levenshtein
from collections import defaultdict
import re
import unicodedata


class SimilitudTextos:
    """Clase que implementa diferentes algoritmos de similitud de textos."""

    def __init__(self):
        """Inicializa stopwords y registro de algoritmos disponibles."""
        self.stop_words = {
            # Español
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
            'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como',
            'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque',
            # Inglés
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her'
        }
        self._init_algoritmos()

    # ---------------- Preprocesamiento ----------------
    def _normalize_text(self, text: str) -> str:
        """Normaliza el texto eliminando acentos y caracteres especiales."""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = text.lower()
        text = re.sub(r"[^a-zñáéíóú\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _preprocess_text(self, text: str) -> List[str]:
        """Tokeniza y limpia texto eliminando stopwords y tokens cortos."""
        text = self._normalize_text(text)
        tokens = [t for t in text.split() if len(t) > 2 and t not in self.stop_words]
        return tokens

    # ---------------- Algoritmos básicos ----------------
    @staticmethod
    def levenshtein(texto1: str, texto2: str) -> float:
        """Distancia de Levenshtein normalizada a similitud [0,1]."""
        if not texto1 or not texto2:
            return 0.0
        distancia = Levenshtein.distance(texto1.lower(), texto2.lower())
        max_len = max(len(texto1), len(texto2))
        return 1 - (distancia / max_len) if max_len else 1.0

    @staticmethod
    def jaccard(texto1: str, texto2: str) -> float:
        """Coeficiente de similitud de Jaccard."""
        if not texto1 or not texto2:
            return 0.0
        set1, set2 = set(texto1.lower().split()), set(texto2.lower().split())
        union = len(set1.union(set2))
        return len(set1.intersection(set2)) / union if union else 0.0

    @staticmethod
    def tfidf_coseno(texto1: str, texto2: str) -> float:
        """Similitud coseno usando TF-IDF."""
        if not texto1 or not texto2:
            return 0.0
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([texto1.lower(), texto2.lower()])
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError:
            # vocabulario vacío
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def ngramas(texto1: str, texto2: str, n: int = 3) -> float:
        """Similitud basada en n-gramas de caracteres."""
        if not texto1 or not texto2:
            return 0.0

        def get_ngrams(text: str, n: int) -> set:
            text = text.lower()
            return {text[i:i+n] for i in range(len(text) - n + 1)}

        ngrams1, ngrams2 = get_ngrams(texto1, n), get_ngrams(texto2, n)
        interseccion = len(ngrams1 & ngrams2)
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        return 2 * interseccion / (len(ngrams1) + len(ngrams2))

    # ---------------- Algoritmos semánticos ----------------
    def semantic_embedding(self, texto1: str, texto2: str, n_components: int = 100) -> float:
        """Similitud semántica usando LSA sobre TF-IDF (aproximación ligera a BERT)."""
        if not texto1 or not texto2:
            return 0.0
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=self._preprocess_text,
                stop_words=list(self.stop_words)
            )
            tfidf = vectorizer.fit_transform([texto1, texto2])
            if tfidf.shape[1] < 2:
                return 0.0
            lsa = TruncatedSVD(n_components=min(n_components, tfidf.shape[1] - 1))
            embeddings = lsa.fit_transform(tfidf)
            return float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0])
        except Exception:
            return 0.0

    def contextual_similarity(self, texto1: str, texto2: str, window: int = 5) -> float:
        """Similitud contextual basada en co-ocurrencia de palabras (estilo Doc2Vec)."""
        if not texto1 or not texto2:
            return 0.0
        try:
            tokens1, tokens2 = self._preprocess_text(texto1), self._preprocess_text(texto2)

            def context_vector(tokens: List[str]) -> Dict[str, float]:
                contexts = defaultdict(float)
                for i, token in enumerate(tokens):
                    for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                        if i != j:
                            contexts[tokens[j]] += 1.0 / abs(i - j)
                return contexts

            ctx1, ctx2 = context_vector(tokens1), context_vector(tokens2)
            comunes = set(ctx1.keys()) & set(ctx2.keys())
            if not comunes:
                return 0.0

            num = sum(ctx1[w] * ctx2[w] for w in comunes)
            norm1 = np.sqrt(sum(v*v for v in ctx1.values()))
            norm2 = np.sqrt(sum(v*v for v in ctx2.values()))
            return float(num / (norm1 * norm2)) if norm1 and norm2 else 0.0
        except Exception:
            return 0.0

    # ---------------- Gestión interna ----------------
    def _init_algoritmos(self):
        """Inicializa el diccionario de algoritmos disponibles."""
        self.ALGORITMOS: Dict[str, Callable[[str, str], float]] = {
            "levenshtein": self.__class__.levenshtein,
            "jaccard": self.__class__.jaccard,
            "tfidf_coseno": self.__class__.tfidf_coseno,
            "ngramas": self.__class__.ngramas,
            "semantic": self.semantic_embedding,
            "contextual": self.contextual_similarity,
        }

    def calcular_similitud(self, texto1: str, texto2: str, algoritmo: str = "tfidf_coseno") -> float:
        """Calcula la similitud entre dos textos usando el algoritmo especificado."""
        if algoritmo not in self.ALGORITMOS:
            raise ValueError(f"Algoritmo '{algoritmo}' no implementado")
        return self.ALGORITMOS[algoritmo](texto1, texto2)


# ---------------- Modo prueba ----------------
def main():
    """Prueba rápida de todos los algoritmos con ejemplos."""
    texto1 = "Los algoritmos de machine learning son fundamentales para el análisis de datos."
    texto2 = "El aprendizaje automático es esencial en el análisis de grandes volúmenes de datos."

    sim = SimilitudTextos()
    for nombre in sim.ALGORITMOS.keys():
        try:
            valor = sim.calcular_similitud(texto1, texto2, nombre)
            print(f"{nombre:<15} -> {valor:.4f}")
        except Exception as e:
            print(f"Error en {nombre}: {e}")


if __name__ == "__main__":
    main()
