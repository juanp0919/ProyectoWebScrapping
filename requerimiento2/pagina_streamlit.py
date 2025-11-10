import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import io
import json

from requerimiento2.similitud_textos import SimilitudTextos
from requerimiento2.analisis_similitud import AnalizadorSimilitud


def mostrar_pagina_similitud():
    """Página de Streamlit para análisis de similitud de textos."""
    
    st.title("📘 Análisis de Similitud de Textos")

    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    modo = st.sidebar.selectbox(
        "Modo de análisis",
        ["Comparar Textos", "Analizar Abstracts"],
        help="Seleccione si quiere comparar textos personalizados o analizar abstracts del BibTeX"
    )
    
    # Instancia temporal para acceder a algoritmos
    sim_temp = SimilitudTextos()
    algoritmo = st.sidebar.selectbox(
        "Algoritmo de similitud",
        list(sim_temp.ALGORITMOS.keys()),
        help="Seleccione el algoritmo para calcular similitud"
    )

    # ---------------- MODO 1: COMPARACIÓN MANUAL ----------------
    if modo == "Comparar Textos":
        st.header("📝 Comparación de Textos")
        
        col1, col2 = st.columns(2)
        with col1:
            texto1 = st.text_area("Texto 1", height=200, placeholder="Ingrese el primer texto...")
        with col2:
            texto2 = st.text_area("Texto 2", height=200, placeholder="Ingrese el segundo texto...")

        if texto1 and texto2:
            similitud = SimilitudTextos()
            with st.spinner("Calculando similitud..."):
                valor = similitud.calcular_similitud(texto1, texto2, algoritmo)
            
            st.metric(label=f"Similitud usando {algoritmo}", value=f"{valor:.4f}")
            st.progress(valor)
        else:
            st.info("Ingrese ambos textos para calcular similitud.")

    # ---------------- MODO 2: ANÁLISIS DE ABSTRACTS ----------------
    else:
        st.header("📊 Análisis de Abstracts desde BibTeX")

        ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
        if not ruta_bib.exists():
            st.error(f"No se encontró el archivo BibTeX en:\n{ruta_bib}")
            return
        
        limite = st.slider(
            "Número de abstracts a analizar",
            min_value=2,
            max_value=20,
            value=5,
            help="Seleccione cuántos abstracts comparar (más abstracts = más tiempo de procesamiento)"
        )

        if st.button("🔍 Analizar Abstracts"):
            try:
                analizador = AnalizadorSimilitud(ruta_bib)
                with st.spinner(f"Ejecutando análisis con '{algoritmo}'..."):
                    matriz = analizador.analizar_abstracts(algoritmo=algoritmo, limite=limite)

                st.success(f"✅ Análisis completado ({matriz.shape[0]} documentos)")
                
                # Mostrar matriz
                st.subheader("Matriz de Similitud")
                st.dataframe(matriz)

                # Heatmap
                st.subheader("Mapa de Calor")
                fig = px.imshow(
                    matriz,
                    labels=dict(x="Documento", y="Documento", color="Similitud"),
                    color_continuous_scale="RdYlBu_r",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Descarga de resultados
                csv_data = matriz.to_csv().encode("utf-8")
                json_data = json.dumps(matriz.to_dict(), indent=2).encode("utf-8")

                st.download_button(
                    label="💾 Descargar CSV",
                    data=csv_data,
                    file_name=f"similitud_{algoritmo}.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="💾 Descargar JSON",
                    data=json_data,
                    file_name=f"similitud_{algoritmo}.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Ocurrió un error al analizar los abstracts: {e}")


if __name__ == "__main__":
    mostrar_pagina_similitud()
