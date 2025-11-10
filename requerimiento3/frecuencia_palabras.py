# requerimiento3/frecuencia_palabras.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Requerimiento 3
Módulo para análisis de frecuencia de palabras/frases en abstracts
para la categoría: "Concepts of Generative AI in Education".

Incluye:
- Conteo de términos de referencia (multi-palabra, ES/EN, variantes con guion).
- Extracción de nuevas frases con TF-IDF (máx. 15 sugeridas).
- Evaluación de precisión/recall/F1 frente a la lista de referencia.
- Exportación de resultados (CSV/JSON) mediante utils.exportador.

Dependencias esperadas: numpy, pandas, scikit-learn, unidecode.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Dict, Set, Optional
from pathlib import Path
import re
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode

from utils.extractor_bib import ExtratorBibTeX
from utils.preprocesador_texto import PreprocesadorTexto
from utils.exportador import Exportador


# ---------------- Utilidades de normalización ---------------- #

def _norm(s: str) -> str:
    """Normaliza: minúsculas, sin tildes, sin dobles espacios."""
    s = unidecode(s.lower().strip())
    s = re.sub(r"\s+", " ", s)
    return s


def _phrase_variants(phrase: str) -> List[str]:
    """
    Genera variantes típicas de una frase normalizada:
    - con espacio vs. guion (fine tuning <-> fine-tuning)
    - versión compacta (sin símbolos no alfanum)
    """
    base = _norm(phrase)
    with_dash = base.replace(" ", "-")
    without_dash = base.replace("-", " ")
    compact = re.sub(r"[^a-z0-9\s-]", "", base)
    return list({base, with_dash, without_dash, compact})


# ======================== Clase principal ======================== #

class AnalizadorFrecuencia:
    """Análisis de frecuencia de términos/frases en abstracts académicos."""

    # ---- LISTA OFICIAL (Requerimiento 3) ----
    PALABRAS_INTERES = {
        "Generative models",
        "Prompting",
        "Machine learning",
        "Multimodality",
        "Fine-tuning",
        "Training data",
        "Algorithmic bias",
        "Explainability",
        "Transparency",
        "Ethics",
        "Privacy",
        "Personalization",
        "Human-AI interaction",
        "AI literacy",
        "Co-creation",
    }

    # Normalizaciones/alias simples para mejorar el conteo
    SINONIMOS: Dict[str, Set[str]] = {
        "generative models": {"generative model", "generative ai models"},
        "prompting": {"prompts", "prompt engineering"},
        "machine learning": {"ml"},
        "multimodality": {"multimodal"},
        "fine-tuning": {"fine tuning", "finetuning"},
        "training data": {"dataset", "datasets"},
        "algorithmic bias": {"bias"},
        "explainability": {"xai", "explainable ai"},
        "transparency": set(),
        "ethics": {"ethical"},
        "privacy": set(),
        "personalization": {"personalisation", "personalized", "personalised"},
        "human-ai interaction": {"human ai interaction", "human computer interaction"},
        "ai literacy": {"ai-illiteracy", "ailiteracy"},
        "co-creation": {"cocreation", "co creation"},
    }

    def __init__(self, ruta_bib: Optional[str] = None):
        """
        Args:
            ruta_bib: Ruta al archivo BibTeX (opcional)
        """
        self.ruta_bib = str(ruta_bib) if ruta_bib else None
        self.preprocesador = PreprocesadorTexto()
        self._abstracts: List[Dict[str, str]] = []

        if ruta_bib:
            self._cargar_abstracts()

        # set con versiones normalizadas de la lista de referencia
        self._referencia_norm: Set[str] = {_norm(p) for p in self.PALABRAS_INTERES}

    # ---------------- Carga abstracts ---------------- #

    def _cargar_abstracts(self) -> None:
        """Carga los abstracts desde el archivo BibTeX."""
        extractor = ExtratorBibTeX(self.ruta_bib)
        self._abstracts = extractor.get_abstracts()

    # ---------------- Conteo términos de referencia ---------------- #

    def _contar_frase_en_texto(self, text_norm: str, frase: str) -> int:
        """
        Cuenta apariciones de 'frase' en 'text_norm' (ambos normalizados).
        Usa patrón de palabra completa, tolerando espacio/guion.
        """
        parts = re.split(r"\s|-", _norm(frase))
        parts = [re.escape(p) for p in parts if p]
        if not parts:
            return 0
        pattern = r"\b" + r"(?:\s|-)+".join(parts) + r"\b"
        return len(re.findall(pattern, text_norm))

    def contar_palabras_predefinidas(self) -> pd.DataFrame:
        """
        Cuenta la frecuencia de las palabras/frases de referencia en los abstracts.

        Returns:
            DataFrame con columnas:
            ['palabra', 'frecuencia_total', 'documentos', 'porcentaje_docs']
        """
        if not self._abstracts:
            raise ValueError("No hay abstracts cargados")

        total_docs = len(self._abstracts)
        frecuencias = Counter()
        docs_por_palabra = Counter()

        # Pre-normalizar todos los abstracts
        textos_norm = [_norm(abstract.get("abstract", "")) for abstract in self._abstracts]

        for frase in sorted(self.PALABRAS_INTERES):
            frase_norm = _norm(frase)

            # ⚠️ FIX: unimos como sets (antes era lista | set -> error)
            variantes = list(
                set(_phrase_variants(frase_norm)) | self.SINONIMOS.get(frase_norm, set())
            )

            doc_hits = 0
            total_hits = 0
            for t in textos_norm:
                hits_doc = 0
                for v in variantes:
                    hits_doc += self._contar_frase_en_texto(t, v)
                if hits_doc > 0:
                    doc_hits += 1
                    total_hits += hits_doc

            frecuencias[frase] = total_hits
            docs_por_palabra[frase] = doc_hits

        resultados = []
        for frase in sorted(self.PALABRAS_INTERES):
            frec = int(frecuencias[frase])
            docs = int(docs_por_palabra[frase])
            resultados.append(
                {
                    "palabra": frase,
                    "frecuencia_total": frec,
                    "documentos": docs,
                    "porcentaje_docs": (docs / total_docs) * 100.0 if total_docs else 0.0,
                }
            )
        df = pd.DataFrame(resultados)
        return df.sort_values(["frecuencia_total", "documentos"], ascending=False)

    # ---------------- Extracción TF-IDF (nuevas frases) ---------------- #

    def extraer_palabras_tfidf(
        self,
        n_palabras: int = 15,
        min_df: float = 0.03,
        incluir_bigramas: bool = True,
        max_features: int = 2000,
    ) -> pd.DataFrame:
        """
        Extrae frases/terminos candidatos con TF-IDF.

        Args:
            n_palabras: número máximo de candidatos a devolver
            min_df: frecuencia mínima de documento (0-1)
            incluir_bigramas: si True, usa ngram_range=(1,2)
            max_features: límite de vocabulario TF-IDF
        """
        if not self._abstracts:
            raise ValueError("No hay abstracts cargados")

        textos = [abstract.get("abstract", "") for abstract in self._abstracts]

        def tokenizer(texto: str) -> List[str]:
            # reusa preprocesador (maneja ES/EN y stopwords); puede devolver uni/bigramas
            return self.preprocesador.procesar_texto(
                texto,
                incluir_bigramas=incluir_bigramas,
            )

        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            ngram_range=(1, 2) if incluir_bigramas else (1, 1),
            min_df=min_df,
            max_df=0.95,
            max_features=max_features,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            lowercase=False,  # ya normalizamos en tokenizer
        )

        X = vectorizer.fit_transform(textos)
        vocab = vectorizer.get_feature_names_out()

        # score promedio ponderado por longitud de documento
        doc_len = X.sum(axis=1).A1 + 1e-9
        weights = doc_len / doc_len.sum()
        scores = np.average(X.toarray(), axis=0, weights=weights)

        df = pd.DataFrame(
            {
                "palabra": vocab,
                "score_tfidf": scores,
                "docs_aparicion": (X > 0).sum(axis=0).A1,
                "freq_relativa": (X > 0).sum(axis=0).A1 / len(textos),
            }
        )

        # quitar candidatas ya presentes en la lista de referencia
        df["pal_norm"] = df["palabra"].map(_norm)
        df = df[~df["pal_norm"].isin(self._referencia_norm)]

        df["score_combinado"] = 0.7 * df["score_tfidf"] + 0.3 * df["freq_relativa"]
        df = df.sort_values("score_combinado", ascending=False).head(n_palabras)
        return df.drop(columns=["pal_norm"])

    # ---------------- Evaluación de precisión ---------------- #

    def calcular_metricas_precision(
        self,
        palabras_extraidas: Set[str],
        tolerancia: float = 0.80,
    ) -> Dict[str, float]:
        """
        Evalúa qué tan “precisas” son las nuevas frases con respecto a la lista
        de referencia, usando coincidencia exacta o similitud difusa.
        """
        if not palabras_extraidas:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        extra_norm = {_norm(p) for p in palabras_extraidas}
        ref = set(self._referencia_norm)

        def is_match(candidate: str) -> bool:
            if candidate in ref:
                return True
            best = max(
                (difflib.SequenceMatcher(None, candidate, r).ratio() for r in ref),
                default=0.0,
            )
            return best >= tolerancia

        tp = sum(1 for c in extra_norm if is_match(c))
        fp = max(len(extra_norm) - tp, 0)
        fn = max(len(ref) - tp, 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {"precision": precision, "recall": recall, "f1_score": f1}

    # ---------------- Exportación ---------------- #

    def exportar_resultados(
        self,
        df_frecuencias: pd.DataFrame,
        df_tfidf: pd.DataFrame,
        metricas: Dict[str, float],
        directorio: str,
    ) -> Dict[str, str]:
        """
        Exporta resultados del análisis (CSV + JSON).
        """
        dir_salida = Path(directorio)
        dir_salida.mkdir(parents=True, exist_ok=True)

        archivos = {}

        # CSV frecuencias
        ruta_csv = dir_salida / "frecuencias_palabras.csv"
        archivos["frecuencias_csv"] = str(Exportador.a_csv(df_frecuencias, ruta_csv))

        # JSON TF-IDF + métricas
        datos_json = {
            "palabras_tfidf": df_tfidf.to_dict(orient="records"),
            "metricas_precision": metricas,
        }
        ruta_json = dir_salida / "analisis_palabras.json"
        archivos["analisis_json"] = str(Exportador.a_json(datos_json, ruta_json))

        return archivos


# ---------------------- Modo prueba CLI opcional ---------------------- #

def main():
    """Pequeña prueba desde CLI."""
    ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
    try:
        analizador = AnalizadorFrecuencia(str(ruta_bib))

        print("\nAnálisis de términos de referencia:")
        df_frecuencias = analizador.contar_palabras_predefinidas()
        print(df_frecuencias.to_string(index=False))

        print("\nCandidatas TF-IDF (nuevas):")
        df_tfidf = analizador.extraer_palabras_tfidf(n_palabras=15, min_df=0.03, incluir_bigramas=True)
        print(df_tfidf.to_string(index=False))

        palabras_extraidas = set(df_tfidf["palabra"].values)
        metricas = analizador.calcular_metricas_precision(palabras_extraidas, tolerancia=0.80)
        print("\nMétricas de precisión:")
        for k, v in metricas.items():
            print(f"{k}: {v:.4f}")

        dir_salida = Path(__file__).parent / "resultados"
        archivos = analizador.exportar_resultados(df_frecuencias, df_tfidf, metricas, str(dir_salida))
        print("\nArchivos generados:")
        for tipo, ruta in archivos.items():
            print(f"- {tipo}: {ruta}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
