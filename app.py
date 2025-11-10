#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
App unificada (sin Seguimiento 2):
- Req. 1 y 2: Scraping + unificación
- Similitud textual (descarga CSV)
- Req. 3: Frecuencia / TF-IDF (descarga CSV)
- Req. 4: Dendrogramas (single/complete/average)
- Req. 5: Visual (mapa por país, nube, líneas temporales + exportar PDF)
"""

from pathlib import Path
from datetime import datetime

import bibtexparser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------- módulos propios -----------------
from requerimiento1.acm_descarga import ACMDescarga
from requerimiento1.sciencedirect import ScienceDirectDescarga
from requerimiento1.unir_bib_deduplicado import UnificadorBibTeX

from requerimiento2.similitud_textos import SimilitudTextos
from requerimiento3.frecuencia_palabras import AnalizadorFrecuencia

# *** Req. 4 ***
from requerimiento4.dendogramas import (
    generar_dendrogramas_desde_bib,
    recomendar_metodo,
)

# *** Req. 5 ***
from requerimiento5.req5_visual import render_req5  # acepta parámetro bib_path opcional

# ---------- Config global ----------
DEFAULT_BIB_PATH = Path("requerimiento1/descargas/resultado_unificado.bib")

st.set_page_config(
    page_title="Análisis Bibliométrico UQ — App Unificada",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estados para no perder resultados entre reruns
for key, default in [
    ("REQ4_RES", None),    # resultados req4
    ("SIM_RESULTS", None), # df con resultados de similitud
    ("SIM_META", None),    # metadatos (IDs/títulos)
    ("FREQ_FREQ_DF", None),# df frecuencias
    ("FREQ_TFIDF_DF", None)# df tfidf
]:
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_data
def cargar_bibliografia(bib_path: Path = DEFAULT_BIB_PATH):
    """Carga el archivo BibTeX unificado."""
    if not bib_path.exists():
        return None
    with open(bib_path, "r", encoding="utf-8") as archivo:
        parser = bibtexparser.bparser.BibTexParser()
        return bibtexparser.load(archivo, parser=parser)


def verificar_archivos():
    """Verifica la existencia de archivos y retorna estado."""
    archivos = {
        "ACM": Path("requerimiento1/descargas/acm/acmCompleto.bib"),
        "ScienceDirect": Path("requerimiento1/descargas/ScienceDirect/sciencedirectCompleto.bib"),
        "Unificado": DEFAULT_BIB_PATH,
    }
    return {nombre: ruta.exists() for nombre, ruta in archivos.items()}


def explicar_algoritmo(nombre):
    """Explicaciones cortas de métodos (tab Similitud)."""
    explicaciones = {
        "Levenshtein": "Número mínimo de ediciones para transformar una cadena en otra.",
        "Jaccard": "Similitud entre conjuntos: |A ∩ B| / |A ∪ B|.",
        "TF-IDF Coseno": "TF-IDF para ponderar términos + similitud coseno de vectores.",
        "N-gramas": "Compara textos por substrings consecutivos.",
        "Semantic Embedding": "Vectores semánticos; coseno en espacio latente.",
        "Contextual Similarity": "Similitud ponderada considerando contexto local.",
    }
    return explicaciones.get(nombre, "Explicación no disponible")


@st.cache_data
def analizar_similitud_articulos(texto1, texto2, metodo):
    similitud = SimilitudTextos()
    metodos = {
        "Levenshtein": similitud.levenshtein,
        "Jaccard": similitud.jaccard,
        "TF-IDF Coseno": similitud.tfidf_coseno,
        "N-gramas": similitud.ngramas,
        "Semantic Embedding": similitud.semantic_embedding,
        "Contextual Similarity": similitud.contextual_similarity,
    }
    return metodos[metodo](texto1, texto2)


def visualizar_similitud(score, metodo):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score) * 100.0,
            title={"text": f"Similitud {metodo}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "lightgray"},
                    {"range": [33, 66], "color": "gray"},
                    {"range": [66, 100], "color": "darkgray"},
                ],
            },
        )
    )
    return fig


def ejecutar_scraping(fuente):
    if fuente == "ACM":
        with st.spinner("Extrayendo referencias de ACM..."):
            extractor = ACMDescarga()
            try:
                extractor.abrir_base_datos()
                return True
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return False
            finally:
                extractor.cerrar()
    elif fuente == "ScienceDirect":
        with st.spinner("Extrayendo referencias de ScienceDirect..."):
            extractor = ScienceDirectDescarga()
            try:
                extractor.abrir_base_datos()
                return True
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return False
            finally:
                extractor.cerrar()
    return False


def make_csv_download(df: pd.DataFrame, filename_prefix: str, button_label: str, help_text: str = ""):
    """Botón de descarga CSV."""
    if df is None or df.empty:
        st.warning("No hay resultados para descargar aún.")
        return
    csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        label=button_label,
        data=csv_bytes,
        file_name=f"{filename_prefix}_{timestamp}.csv",
        mime="text/csv",
        help=help_text,
        use_container_width=True,
    )


def mostrar_seccion_scraping():
    st.header("🌐 Web Scraping de Referencias")
    estados = verificar_archivos()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ACM Digital Library")
        boton_acm = st.button("Extraer de ACM", disabled=estados["ACM"])
        if estados["ACM"]:
            st.success("✅ Referencias de ACM ya extraídas")
        if boton_acm and ejecutar_scraping("ACM"):
            st.success("✅ Referencias extraídas exitosamente")

    with col2:
        st.subheader("ScienceDirect")
        boton_sd = st.button("Extraer de ScienceDirect", disabled=estados["ScienceDirect"])
        if estados["ScienceDirect"]:
            st.success("✅ Referencias de ScienceDirect ya extraídas")
        if boton_sd and ejecutar_scraping("ScienceDirect"):
            st.success("✅ Referencias extraídas exitosamente")

    st.subheader("Unificación de Referencias")
    boton_unificar = st.button(
        "Unificar Referencias",
        disabled=estados["Unificado"] or not (estados["ACM"] and estados["ScienceDirect"]),
    )
    if estados["Unificado"]:
        st.success("✅ Referencias ya unificadas")
    elif boton_unificar:
        with st.spinner("Unificando referencias..."):
            unificador = UnificadorBibTeX()
            try:
                if unificador.unificar():
                    st.success("✅ Referencias unificadas exitosamente")
                else:
                    st.error("❌ Error al unificar")
            except Exception as e:
                st.error(f"❌ {e}")


# ==============================  MAIN  ==============================

def main():
    st.title("📚 Sistema de Análisis Bibliométrico UQ — App Unificada")
    st.markdown(
        """
Sistema integrado para análisis bibliométrico que incluye:
- Web scraping y unificación  
- Similitud textual (con descarga CSV)  
- Frecuencia/TF-IDF (con descarga CSV)  
- **Clustering jerárquico (dendrogramas)**  
- **Visualizaciones geográfica/temporal (Req. 5)**
"""
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🌐 Web Scraping",
            "🔄 Similitud",
            "📊 Frecuencia (Req. 3)",
            "🔬 Dendrogramas (Req. 4)",
            "🌎 Visual (Req. 5)",
        ]
    )

    # 1) Web scraping
    with tab1:
        mostrar_seccion_scraping()

    # 2) Similitud
    with tab2:
        st.header("🔄 Análisis de Similitud")
        bibliografia = cargar_bibliografia(DEFAULT_BIB_PATH)
        if bibliografia is None:
            st.error("❌ No se encontró el archivo unificado")
        else:
            articulos = [
                (entry.get("ID", ""), entry.get("title", ""))
                for entry in bibliografia.entries if "abstract" in entry
            ]
            c1, c2 = st.columns(2)
            with c1:
                art1 = st.selectbox("Primer artículo", options=articulos, format_func=lambda x: x[1])
            with c2:
                art2 = st.selectbox("Segundo artículo", options=articulos, format_func=lambda x: x[1])

            resultados = {}
            if art1 and art2:
                abstract1 = next(e["abstract"] for e in bibliografia.entries if e["ID"] == art1[0])
                abstract2 = next(e["abstract"] for e in bibliografia.entries if e["ID"] == art2[0])

                with st.expander("Ver Abstracts"):
                    st.markdown("**Abstract 1:**")
                    st.write(abstract1)
                    st.markdown("**Abstract 2:**")
                    st.write(abstract2)

                metodos = [
                    "Levenshtein",
                    "Jaccard",
                    "TF-IDF Coseno",
                    "N-gramas",
                    "Semantic Embedding",
                    "Contextual Similarity",
                ]
                method_tabs = st.tabs(metodos)

                for metodo, tb in zip(metodos, method_tabs):
                    with tb:
                        a, b = st.columns([1, 1])
                        with a:
                            st.markdown(explicar_algoritmo(metodo))
                        with b:
                            with st.spinner(f"Calculando {metodo}..."):
                                score = analizar_similitud_articulos(abstract1, abstract2, metodo)
                                resultados[metodo] = score
                                st.plotly_chart(visualizar_similitud(score, metodo), use_container_width=True)

                st.header("📊 Comparativa de Métodos")
                categorias = list(resultados.keys())
                valores = list(resultados.values())
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=valores, theta=categorias, fill="toself", name="Similitud"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Descarga CSV
                sim_df = pd.DataFrame(
                    {
                        "id_articulo_1": [art1[0]] * len(resultados),
                        "titulo_articulo_1": [art1[1]] * len(resultados),
                        "id_articulo_2": [art2[0]] * len(resultados),
                        "titulo_articulo_2": [art2[1]] * len(resultados),
                        "metodo": list(resultados.keys()),
                        "score": list(map(float, resultados.values())),
                    }
                )
                st.session_state["SIM_RESULTS"] = sim_df
                st.session_state["SIM_META"] = {
                    "id1": art1[0], "title1": art1[1],
                    "id2": art2[0], "title2": art2[1]
                }

                st.subheader("⬇️ Descargar resultados de similitud")
                make_csv_download(
                    df=sim_df,
                    filename_prefix="similitud_articulos",
                    button_label="Descargar CSV de similitud",
                    help_text="Incluye IDs, títulos y score por método.",
                )

    # 3) Frecuencia / TF-IDF (Req. 3)
    with tab3:
        st.header("📊 Análisis de Frecuencia (Req. 3)")
        if not DEFAULT_BIB_PATH.exists():
            st.error("❌ No se encontró el archivo unificado. Primero realice el scraping.")
        else:
            try:
                analizador = AnalizadorFrecuencia(str(DEFAULT_BIB_PATH))
                c1, c2 = st.columns(2)
                with c1:
                    min_df = st.slider("Frecuencia mínima de documento", 0.01, 0.5, 0.05, 0.01)
                with c2:
                    incluir_bigramas = st.checkbox("Incluir bigramas", value=True)

                if st.button("Realizar Análisis", key="analisis_frecuencia"):
                    with st.spinner("Analizando textos..."):
                        df_freq = analizador.contar_palabras_predefinidas()
                        df_tfidf = analizador.extraer_palabras_tfidf(
                            min_df=min_df, incluir_bigramas=incluir_bigramas
                        )

                        st.session_state["FREQ_FREQ_DF"] = df_freq.copy()
                        st.session_state["FREQ_TFIDF_DF"] = df_tfidf.copy()

                        st.subheader("📈 Palabras Predefinidas")
                        a, b = st.columns(2)
                        with a:
                            fig_freq = px.bar(
                                df_freq.head(10), x="palabra", y="frecuencia_total",
                                title="Top 10 Frecuencia Total"
                            )
                            fig_freq.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_freq, use_container_width=True)
                        with b:
                            fig_docs = px.bar(
                                df_freq.head(10), x="palabra", y="porcentaje_docs",
                                title="Cobertura en Documentos"
                            )
                            fig_docs.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_docs, use_container_width=True)

                        st.subheader("🔍 TF-IDF")
                        a, b = st.columns(2)
                        with a:
                            fig_tfidf = px.bar(
                                df_tfidf.head(10), x="palabra", y="score_tfidf",
                                title="Top 10 por TF-IDF"
                            )
                            fig_tfidf.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_tfidf, use_container_width=True)
                        with b:
                            fig_rel = px.pie(
                                df_tfidf.head(10), values="freq_relativa", names="palabra",
                                title="Distribución frecuencia relativa"
                            )
                            st.plotly_chart(fig_rel, use_container_width=True)

                        st.markdown("##### Detalle TF-IDF")
                        st.dataframe(
                            df_tfidf.style.format(
                                {"score_tfidf": "{:.4f}", "freq_relativa": "{:.2%}", "score_combinado": "{:.4f}"}
                            )
                        )

                if st.session_state["FREQ_FREQ_DF"] is not None or st.session_state["FREQ_TFIDF_DF"] is not None:
                    st.markdown("---")
                    st.subheader("⬇️ Descargar resultados")
                    col_csv1, col_csv2 = st.columns(2)
                    with col_csv1:
                        make_csv_download(
                            df=st.session_state["FREQ_FREQ_DF"],
                            filename_prefix="frecuencias_predefinidas",
                            button_label="Descargar CSV (Frecuencias)",
                            help_text="Frecuencia total y % de documentos por palabra monitoreada.",
                        )
                    with col_csv2:
                        make_csv_download(
                            df=st.session_state["FREQ_TFIDF_DF"],
                            filename_prefix="tfidf_resultados",
                            button_label="Descargar CSV (TF-IDF)",
                            help_text="Scores TF-IDF y métricas derivadas por término.",
                        )
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Verifique el formato del .bib y la existencia de abstracts.")

    # 4) Dendrogramas (Req. 4)
    with tab4:
        st.header("🔬 Requerimiento 4 — Clustering jerárquico con dendrogramas")
        if not DEFAULT_BIB_PATH.exists():
            st.error("❌ No se encontró el archivo unificado. Primero complete el scraping y la unificación.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                max_docs = st.slider("Máximo de abstracts a usar", 100, 3000, 1000, 100)
            with c2:
                max_terms = st.slider("Términos TF-IDF (max_features)", 2000, 30000, 10000, 1000)

            # FIX: evitar select_slider con tuplas -> usamos selectbox
            opciones_ngrams = {"1-1": (1, 1), "1-2": (1, 2), "1-3": (1, 3), "2-2": (2, 2)}
            etiqueta = st.selectbox("N-gramas", list(opciones_ngrams.keys()), index=1)
            ngrams = opciones_ngrams[etiqueta]

            if st.button("Construir dendrogramas", key="btn_req4"):
                with st.spinner("Generando dendrogramas..."):
                    res = generar_dendrogramas_desde_bib(
                        bib_path=str(DEFAULT_BIB_PATH),
                        carpeta_salida="requerimiento4/resultados",
                        max_docs=int(max_docs),
                        ngram_range=ngrams,
                        max_features=int(max_terms),
                    )
                    st.session_state["REQ4_RES"] = res

            res = st.session_state["REQ4_RES"]
            if res:
                best = recomendar_metodo(res)
                st.success(f"Método recomendado: **{best.method}** "
                           f"(silhouette={best.silhouette:.4f}, k*={best.n_clusters})")

                cols = st.columns(3)
                for (method, r), col in zip(res.items(), cols*2):  # 3 métodos
                    with col:
                        st.subheader(method.title())
                        st.image(str(r.dendrogram_png), use_column_width=True)
                        st.caption(f"PNG: {r.dendrogram_png.name} • PDF: {r.dendrogram_pdf.name}")
            else:
                st.info("Configura los parámetros y pulsa **Construir dendrogramas**.")

    # 5) Visualizaciones (Req. 5)
    with tab5:
        st.header("🌎 Requerimiento 5 — Visualizaciones geográfica/temporal y nube de palabras")
        render_req5(bib_path=str(DEFAULT_BIB_PATH))


if __name__ == "__main__":
    main()
