# requerimiento5/req5_visual.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Requerimiento 5: Analítica visual de la producción científica

Incluye:
1) Mapa (choropleth) por país del primer autor.
2) Nube de palabras dinámica (abstracts + keywords; ES/EN).
3) Línea temporal de publicaciones por año y por revista.
4) Exportación de las 3 visualizaciones a PDF.

Dependencias:
- pandas, plotly, wordcloud, pycountry, kaleido, reportlab, bibtexparser, Unidecode, pillow
Instala (si falta algo):
pip install pandas plotly wordcloud pycountry kaleido reportlab bibtexparser Unidecode pillow
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict, Any
import re
import io

import pandas as pd
import plotly.express as px
import plotly.io as pio
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import pycountry
import bibtexparser
from unidecode import unidecode

# Exportación PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader

# Config plotly/kaleido (export a PNG)
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1600
pio.kaleido.scope.default_height = 900


# ------------------------ Carga de BibTeX ------------------------

def cargar_bib(bib_path: str) -> Any:
    """Carga un archivo .bib con bibtexparser."""
    p = Path(bib_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo BibTeX: {p}")
    with open(p, "r", encoding="utf-8") as f:
        parser = bibtexparser.bparser.BibTexParser()
        return bibtexparser.load(f, parser=parser)


# ------------------------ Utilidades país ------------------------

def _country_map_variants() -> Dict[str, str]:
    """
    Devuelve un mapa: variante de nombre / código de país (minúsculas)
    -> código ISO-3.

    Incluye:
    - name, official_name, common_name
    - versiones sin acento
    - códigos alpha_2 (us, es, fr)
    - códigos alpha_3 (usa, esp, fra)
    - alias comunes (USA, UK, etc.)
    """
    mapping: Dict[str, str] = {}

    for c in pycountry.countries:
        names = set()

        # Nombres base
        names.add(c.name)
        for attr in ("official_name", "common_name"):
            if hasattr(c, attr):
                names.add(getattr(c, attr))

        # Versiones normalizadas (sin acento, minúsculas)
        norm_names = set()
        for n in names:
            norm_names.add(unidecode(n).lower())

        # Códigos ISO
        norm_names.add(c.alpha_2.lower())
        norm_names.add(c.alpha_3.lower())

        # Alias manuales para algunos países comunes
        if c.alpha_2 == "US":
            norm_names.update({
                "usa", "u.s.a", "united states", "united states of america",
                "estados unidos", "u.s.", "us"
            })
        if c.alpha_2 == "GB":
            norm_names.update({
                "uk", "u.k.", "united kingdom", "great britain",
                "reino unido", "gb"
            })
        if c.alpha_2 == "ES":
            norm_names.update({"spain", "espana", "españa"})
        if c.alpha_2 == "DE":
            norm_names.update({"germany", "alemania"})
        if c.alpha_2 == "FR":
            norm_names.update({"france", "francia"})

        # Rellenar mapping
        for n in norm_names:
            mapping[n] = c.alpha_3

    return mapping


_COUNTRY_MAP = _country_map_variants()


def _safe_year(entry: Dict[str, Any]) -> int | None:
    y = entry.get("year") or entry.get("date")
    if not y:
        return None
    try:
        return int(str(y)[:4])
    except Exception:
        return None


def _infer_pais_primer_autor(entry: Dict[str, Any]) -> str:
    """
    Heurística mejorada:
    1) Busca códigos/nombres de país en campos de afiliación / dirección.
    2) Luego en author / note.
    3) Soporta nombres, alias y códigos ISO-2/ISO-3.

    Retorna ISO-3; 'UNK' si no se detecta.
    """
    # Campos donde es más probable encontrar el país
    primary_fields = (
        "affiliation_country",
        "country",
        "affiliation",
        "affiliations",
        "address",
        "institution",
        "school",
        "organization",
        "publisher",
        "location",
    )
    secondary_fields = ("author", "note")

    def _buscar_en_texto(txt: str) -> str | None:
        if not txt:
            return None
        t = unidecode(txt).lower()

        # 1) Primero por tokens exactos (códigos, nombres cortos)
        tokens = re.split(r"[^a-zA-Z]+", t)
        for tok in tokens:
            if not tok:
                continue
            iso = _COUNTRY_MAP.get(tok)
            if iso:
                return iso

        # 2) Después por nombres compuestos (united states, reino unido…)
        for name_lc, iso3 in _COUNTRY_MAP.items():
            if " " in name_lc and name_lc in t:
                return iso3

        return None

    # Buscar en campos principales
    for key in primary_fields:
        iso = _buscar_en_texto(entry.get(key, ""))
        if iso:
            return iso

    # Buscar en campos secundarios (menos fiables)
    for key in secondary_fields:
        iso = _buscar_en_texto(entry.get(key, ""))
        if iso:
            return iso

    return "UNK"


# ------------------------ 1) Mapa por país ------------------------

def construir_mapa_paises(bib_db: Any):
    """
    Devuelve (fig_plotly, df_conteos) para la distribución por país del primer autor.
    df_conteos: columnas [iso3, count]
    """
    rows = []
    for e in getattr(bib_db, "entries", []):
        iso3 = _infer_pais_primer_autor(e)
        rows.append({"iso3": iso3})
    df = pd.DataFrame(rows)
    if df.empty:
        fig = px.choropleth(
            locations=[],
            locationmode="ISO-3",
            color=[],
            title="Mapa por país (sin datos)"
        )
        return fig, df

    conteos = df.value_counts("iso3").reset_index(name="count")
    conteos = conteos[conteos["iso3"] != "UNK"]  # excluir desconocidos del mapa
    fig = px.choropleth(
        conteos,
        locations="iso3",
        color="count",
        color_continuous_scale="Viridis",
        title="Distribución geográfica del primer autor (por país)",
        locationmode="ISO-3",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig, conteos


# ------------------------ 2) Nube de palabras ------------------------

_ES_STOP = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no",
    "una", "su", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "este", "si", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien", "desde",
}
_EN_MORE = {
    "we", "our", "they", "these", "those", "using", "use", "used", "results", "result", "paper", "study",
    "approach", "based", "within", "across", "among", "show", "shown", "however", "therefore", "method",
    "methods", "conclusion", "conclusions", "introduction", "discussion"
}
STOP_COMBINED = (set(STOPWORDS) | _ES_STOP | _EN_MORE)


def _texto_abstracts_keywords(bib_db: Any) -> str:
    texts: List[str] = []
    for e in getattr(bib_db, "entries", []):
        abs_txt = e.get("abstract", "")
        if abs_txt:
            texts.append(abs_txt)
        keys = e.get("keywords") or e.get("keyword") or ""
        if keys:
            parts = re.split(r"[;,/]", keys)
            texts.extend([p.strip() for p in parts if p.strip()])
    big = " ".join(texts)
    return unidecode(big)


def construir_nube_palabras(bib_db: Any, ancho: int = 1600, alto: int = 900) -> Image.Image:
    """Genera imagen PIL con la nube de palabras (abstracts + keywords)."""
    text = _texto_abstracts_keywords(bib_db)
    if not text.strip():
        return Image.new("RGB", (ancho, alto), (245, 245, 245))
    wc = WordCloud(
        width=ancho,
        height=alto,
        background_color="white",
        stopwords=STOP_COMBINED,
        collocations=True,
        colormap="viridis",
        prefer_horizontal=0.9,
        max_words=500,
    ).generate(text)
    return wc.to_image()


# ------------------------ 3) Línea temporal ------------------------

def construir_timeline(bib_db: Any):
    """
    Devuelve:
      - fig_line_anual: publicaciones por año
      - fig_line_revista: por año y revista (Top 10)
      - df_anual, df_revista: DataFrames agregados
    """
    rows = []
    for e in getattr(bib_db, "entries", []):
        y = _safe_year(e)
        j = e.get("journal") or e.get("booktitle") or e.get("venue") or "Sin revista"
        if y:
            rows.append({"year": y, "journal": j})
    df = pd.DataFrame(rows)

    if df.empty:
        fig1 = px.line(title="Publicaciones por año (sin datos)")
        fig2 = px.line(title="Publicaciones por año y revista (sin datos)")
        return fig1, fig2, pd.DataFrame(), pd.DataFrame()

    df_anual = df.value_counts("year").reset_index(name="count").sort_values("year")
    df_rev = df.value_counts(["year", "journal"]).reset_index(name="count").sort_values(["year", "journal"])

    fig1 = px.line(df_anual, x="year", y="count", markers=True, title="Publicaciones por año")
    fig1.update_layout(xaxis_title="Año", yaxis_title="# publicaciones", margin=dict(l=10, r=10, t=60, b=10))

    # Top 10 revistas por total para legibilidad
    top_revistas = df_rev.groupby("journal")["count"].sum().sort_values(ascending=False).head(10).index
    df_rev_plot = df_rev[df_rev["journal"].isin(top_revistas)]

    fig2 = px.line(
        df_rev_plot,
        x="year",
        y="count",
        color="journal",
        markers=True,
        title="Publicaciones por año y por revista (Top 10 revistas)",
    )
    fig2.update_layout(
        xaxis_title="Año",
        yaxis_title="# publicaciones",
        legend_title="Revista",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig1, fig2, df_anual, df_rev


# ------------------------ 4) Exportación a PDF ------------------------

def _plotly_to_png_bytes(fig) -> bytes:
    return fig.to_image(format="png")


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def exportar_pdf(fig_mapa, img_nube: Image.Image, fig_line1, fig_line2, out_pdf_path: str) -> str:
    """
    Crea un PDF con: mapa, nube de palabras, línea por año, línea por revista.
    Devuelve la ruta del PDF creado.
    """
    out = Path(out_pdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convertir a PNG bytes
    mapa_png = _plotly_to_png_bytes(fig_mapa)
    nube_png = _pil_to_png_bytes(img_nube)
    line1_png = _plotly_to_png_bytes(fig_line1)
    line2_png = _plotly_to_png_bytes(fig_line2)

    # Armar PDF apaisado
    c = canvas.Canvas(str(out), pagesize=landscape(A4))
    W, H = landscape(A4)

    def _draw(title: str, png_bytes: bytes):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, H - 40, title)
        img = ImageReader(io.BytesIO(png_bytes))
        margin = 30
        c.drawImage(
            img,
            margin,
            60,
            width=W - 2 * margin,
            height=H - 120,
            preserveAspectRatio=True,
            anchor="c",
        )
        c.showPage()

    _draw("Mapa por país del primer autor", mapa_png)
    _draw("Nube de palabras (abstracts + keywords)", nube_png)
    _draw("Publicaciones por año", line1_png)
    _draw("Publicaciones por año y revista (Top 10)", line2_png)

    c.save()
    return str(out)


# ------------------------ UI para Streamlit ------------------------

def render_req5(bib_path: str = "requerimiento1/descargas/resultado_unificado.bib") -> None:
    """
    Renderiza el Requerimiento 5 dentro de Streamlit.
    Se usa desde app.py como una pestaña.
    """
    import streamlit as st

    st.markdown(
        "Esta sección muestra **mapa**, **nube de palabras** y **líneas temporales** "
        "a partir del archivo unificado `.bib`."
    )

    # Parámetros simples (puedes ampliar si quieres)
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        ancho = st.slider("Ancho nube de palabras (px)", 800, 2400, 1600, 100)
    with col_cfg2:
        alto = st.slider("Alto nube de palabras (px)", 400, 1400, 900, 50)

    # Carga
    try:
        bib = cargar_bib(bib_path)
    except FileNotFoundError:
        st.error(
            f"No se encontró el archivo BibTeX en `{bib_path}`. "
            "Completa primero el scraping/unificación o ajusta la ruta."
        )
        return

    # Construcciones
    with st.spinner("Generando visualizaciones…"):
        fig_mapa, df_paises = construir_mapa_paises(bib)
        img_nube = construir_nube_palabras(bib, ancho=ancho, alto=alto)
        fig_y, fig_yj, df_anual, df_rev = construir_timeline(bib)

    # Visuales
    st.subheader("Mapa por país del primer autor")
    st.plotly_chart(fig_mapa, use_container_width=True)
    if not df_paises.empty:
        st.dataframe(df_paises.rename(columns={"iso3": "País (ISO-3)", "count": "Conteo"}))

    st.subheader("Nube de palabras (abstracts + keywords)")
    st.image(img_nube, use_column_width=True)

    st.subheader("Publicaciones por año")
    st.plotly_chart(fig_y, use_container_width=True)
    if not df_anual.empty:
        st.dataframe(df_anual.rename(columns={"year": "Año", "count": "Conteo"}))

    st.subheader("Publicaciones por año y por revista (Top 10)")
    st.plotly_chart(fig_yj, use_container_width=True)

    # Exportación a PDF
    st.markdown("---")
    st.subheader("Exportar las 3 visualizaciones a PDF")
    if st.button("🧾 Generar PDF"):
        out_path = "requerimiento5/resultados/req5_visual.pdf"
        path_pdf = exportar_pdf(fig_mapa, img_nube, fig_y, fig_yj, out_path)
        st.success(f"PDF generado: {path_pdf}")
        # Botón de descarga
        with open(path_pdf, "rb") as fh:
            st.download_button(
                label="⬇️ Descargar PDF",
                data=fh.read(),
                file_name="req5_visual.pdf",
                mime="application/pdf",
            )


# ------------------------ Demo CLI opcional ------------------------

def main_demo():
    """Ejecuta el pipeline y crea un PDF de ejemplo."""
    bib = cargar_bib("requerimiento1/descargas/resultado_unificado.bib")

    fig_mapa, _ = construir_mapa_paises(bib)
    img_nube = construir_nube_palabras(bib)
    fig_y, fig_yj, _, _ = construir_timeline(bib)

    out_pdf = exportar_pdf(fig_mapa, img_nube, fig_y, fig_yj, "requerimiento5/resultados/req5_visual.pdf")
    print("PDF generado en:", out_pdf)


if __name__ == "__main__":
    main_demo()
