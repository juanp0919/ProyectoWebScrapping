# -*- coding: utf-8 -*-
"""
Une y deduplica BibTeX de ACM + ScienceDirect en un solo archivo.
Criterio de duplicado: título (normalizado).
Fusión de campos para preservar la mayor cantidad de información.

Changelog:
- Genera `duplicados_eliminados.bib` con las entradas descartadas por duplicado.  # NOTE
- Genera `registros_incompletos.csv` para diagnosticar faltantes (title/year/abstract).  # NOTE
- Mantiene compatibilidad con la clase UnificadorBibTeX.unificar() → bool.
"""

import os
import re
import sys
import csv  # NOTE: para exportar registros incompletos
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple

# ---------------- Rutas ----------------

# Ruta relativa a la carpeta de descargas
BASE_DIR = Path(__file__).parent / "descargas"
# Asegurarse de que las carpetas existan
BASE_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "acm").mkdir(exist_ok=True)
(BASE_DIR / "ScienceDirect").mkdir(exist_ok=True)

# Entradas de origen
ACM_BIB = BASE_DIR / "acm" / "acmCompleto.bib"
SCIENCE_BIB = BASE_DIR / "ScienceDirect" / "sciencedirectCompleto.bib"

# Salidas
SALIDA_BIB = BASE_DIR / "resultado_unificado.bib"
DUPS_BIB = BASE_DIR / "duplicados_eliminados.bib"     # NOTE
INCOMPLETOS_CSV = BASE_DIR / "registros_incompletos.csv"  # NOTE

# ---------------- Clase principal ----------------

class UnificadorBibTeX:
    """Clase para unificar y deduplicar archivos BibTeX."""
    
    def __init__(self):
        """Inicializa el unificador con las rutas predefinidas."""
        self.base_dir = BASE_DIR
        self.acm_bib = ACM_BIB
        self.science_bib = SCIENCE_BIB
        self.salida_bib = SALIDA_BIB
        self.dup_bib = DUPS_BIB                   # NOTE
        self.incompletos_csv = INCOMPLETOS_CSV    # NOTE
    
    def unificar(self) -> bool:
        """
        Unifica y deduplica los archivos BibTeX.
        
        Returns:
            bool: True si el proceso fue exitoso, False en caso contrario
        """
        try:
            print(f"📥 Leyendo:\n - {self.acm_bib}\n - {self.science_bib}")
            acm_entries = cargar_bib(self.acm_bib)
            sd_entries = cargar_bib(self.science_bib)

            all_entries: List[Dict] = []
            all_entries.extend(acm_entries)
            all_entries.extend(sd_entries)

            if not all_entries:
                print("❌ No se encontraron entradas en los .bib de origen.")
                return False

            # Diagnóstico de registros incompletos (no impide unificación)  # NOTE
            incompletos = detectar_incompletos(all_entries)
            if incompletos:
                exportar_incompletos_csv(incompletos, self.incompletos_csv)
                print(f"📝 Registros incompletos listados en: {self.incompletos_csv}")
            else:
                # Si no hay incompletos de interés, elimina CSV previo para no confundir
                try:
                    if self.incompletos_csv.exists():
                        self.incompletos_csv.unlink()
                except Exception:
                    pass

            # Deduplicar por título normalizado
            by_title: Dict[str, Dict] = {}
            duplicados_descartados: List[Dict] = []  # NOTE: recolecta duplicados
            for e in all_entries:
                tkey = normalized_title(e)
                if not tkey:
                    # Sin título → clave estable por ID para no perder el registro
                    tkey = f"__notitle__::{e.get('ID','noid')}"
                if tkey in by_title:
                    # Registrar el que se reemplaza como duplicado/descartado   # NOTE
                    duplicados_descartados.append(e)
                    by_title[tkey] = merge_entries(by_title[tkey], e)
                else:
                    by_title[tkey] = e

            # Reasignar ID seguro
            final_entries: List[Dict] = []
            for idx, (tkey, e) in enumerate(by_title.items(), start=1):
                if not e.get("ID"):
                    base = re.sub(r"\s+", "_", tkey)[:40].strip("_")
                    e["ID"] = f"{base or 'entry'}_{idx}"
                final_entries.append(e)

            print(f"🧮 Entradas originales: {len(all_entries)}")
            print(f"✅ Entradas tras deduplicar: {len(final_entries)}")
            print(f"♻️ Duplicados descartados (registrados): {len(duplicados_descartados)}")  # NOTE

            # Guardar unificado
            guardar_bib(final_entries, self.salida_bib)
            print(f"💾 Archivo unificado escrito en: {self.salida_bib}")

            # Guardar duplicados descartados (si los hay)  # NOTE
            if duplicados_descartados:
                guardar_bib(duplicados_descartados, self.dup_bib)
                print(f"📄 Archivo con duplicados eliminados: {self.dup_bib}")
            else:
                # limpiar archivo anterior si existía
                try:
                    if self.dup_bib.exists():
                        self.dup_bib.unlink()
                except Exception:
                    pass

            return True
            
        except Exception as e:
            print(f"❌ Error al unificar: {e}")
            return False

# ---------------- Utilidades de normalización / split ----------------

PUNCT_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)

def strip_braces(s: str) -> str:
    return s.replace("{", "").replace("}", "")

def normalize_text(s: str) -> str:
    s = strip_braces(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = PUNCT_PATTERN.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalized_title(entry: Dict) -> str:
    """Obtiene el título normalizado usando varias claves posibles."""
    title = entry.get("title") or entry.get("Title") or entry.get("TITLE") or ""
    return normalize_text(title)

def smart_union_list(values, sep_candidates=(" and ", ";", ",")):
    """
    Une listas representadas como string usando separadores típicos en BibTeX.
    Devuelve string con separador '; ' sin duplicados, preservando orden de aparición.
    """
    seen = set()
    ordered = []
    for s in values:
        if s is None:
            continue
        txt = str(s).strip()
        if not txt:
            continue
        parts = [txt]
        # intentar todos los separadores para capturar casos mixtos
        for sep in sep_candidates:
            new_parts = []
            for p in parts:
                new_parts.extend([pp.strip() for pp in p.split(sep) if pp.strip()])
            parts = new_parts
        for p in parts:
            key = normalize_text(p)
            if key not in seen:
                seen.add(key)
                ordered.append(p)
    return "; ".join(ordered)

def merge_scalar(a: str, b: str) -> str:
    """
    Para campos escalares: si uno contiene al otro, usa el más largo.
    Si son distintos y no contenidos, concatena con ' | ' sin duplicar.
    """
    if not a: return b
    if not b: return a
    if a == b: return a
    if a in b: return b
    if b in a: return a
    if normalize_text(a) == normalize_text(b):
        return a if len(a) >= len(b) else b
    return f"{a} | {b}"

# ---------------- bibtexparser (con fallback simple) ----------------

def _load_with_bibtexparser(path: Path):
    import bibtexparser
    with open(path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f)
    return db

def _dump_with_bibtexparser(entries, salida: Path):
    import bibtexparser
    from bibtexparser.bwriter import BibTexWriter
    from bibtexparser.bibdatabase import BibDatabase

    db = BibDatabase()
    db.entries = entries
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = ("ID",)
    with open(salida, "w", encoding="utf-8") as f:
        f.write(writer.write(db))

def _simple_bib_parse(text: str):
    """
    Parser simple de respaldo (no cubre 100% de casos, pero funciona para la mayoría).
    Devuelve lista de dicts con claves: ENTRYTYPE, ID y campos.
    """
    entries = []
    # separar por entradas @
    for block in re.split(r"(?m)^@", text):
        block = block.strip()
        if not block:
            continue
        # ejemplo: ARTICLE{id,
        m = re.match(r"(\w+)\s*\{\s*([^,]+)\s*,(.*)\}\s*$", block, re.DOTALL)
        if not m:
            # intentar otra forma hasta primera llave
            m2 = re.match(r"(\w+)\s*\{\s*([^,]+)\s*,(.*)", block, re.DOTALL)
            if not m2:
                continue
            entrytype, id_, rest = m2.groups()
        else:
            entrytype, id_, rest = m.groups()
        fields = {}
        # partir por líneas field = {value} o field = "value"
        for line in re.split(r",\s*\n", rest):
            kv = line.strip().rstrip(",")
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip().strip(",")
            # quitar braces o comillas envolventes
            if (v.startswith("{") and v.endswith("}")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1].strip()
            if k:
                fields[k.lower()] = v
        entry = {"ENTRYTYPE": entrytype.lower(), "ID": id_.strip()}
        entry.update(fields)
        entries.append(entry)
    return entries

def _simple_bib_dump(entries, salida: Path):
    def fmt_field(k, v):
        return f"  {k} = {{{v}}}"
    lines = []
    for e in entries:
        et = e.get("ENTRYTYPE", "article")
        id_ = e.get("ID", "noid")
        # ordenar campos dejando ENTRYTYPE e ID fuera
        body_fields = [(k, v) for k, v in e.items() if k not in ("ENTRYTYPE", "ID")]
        body = ",\n".join(fmt_field(k, v) for k, v in body_fields if v)
        block = f"@{et}{{{id_},\n{body}\n}}\n"
        lines.append(block)
    salida.write_text("".join(lines), encoding="utf-8")

def cargar_bib(path: Path):
    if not path.exists():
        print(f"⚠ No existe: {path}")
        return []
    try:
        db = _load_with_bibtexparser(path)
        return db.entries  # type: ignore[attr-defined]
    except Exception as e:
        print(f"ℹ bibtexparser no disponible o falló ({e}). Usando parser simple.")
        text = path.read_text(encoding="utf-8")
        return _simple_bib_parse(text)

def guardar_bib(entries, salida: Path):
    try:
        _dump_with_bibtexparser(entries, salida)
    except Exception as e:
        print(f"ℹ No pude usar bibtexparser para escribir ({e}). Usando volcado simple.")
        _simple_bib_dump(entries, salida)

# ---------------- Lógica de fusión / deduplicación ----------------

LIST_FIELDS = {
    "author": (" and ", ";", ","),
    "keywords": (";", ","),
}
UNION_FIELDS = {"author", "keywords", "doi", "url"}  # doi/url también se unen sin repetir

def merge_entries(e1: Dict, e2: Dict) -> Dict:
    merged = dict(e1)  # base
    # Alinear ENTRYTYPE
    et1 = e1.get("ENTRYTYPE", "")
    et2 = e2.get("ENTRYTYPE", "")
    if et1 and et2 and et1 != et2:
        merged.setdefault("entrytype_alt", et2)
    elif not et1 and et2:
        merged["ENTRYTYPE"] = et2

    # IDs alternativos (para rastreabilidad)
    aliases = []
    if "ID" in e1: aliases.append(str(e1["ID"]))
    if "ID" in e2 and e2["ID"] != e1.get("ID"): aliases.append(str(e2["ID"]))
    if aliases:
        merged["aliases"] = smart_union_list(["; ".join(aliases)], sep_candidates=(";",))

    # Fusionar campos
    keys = set(e1.keys()) | set(e2.keys())
    for k in keys:
        if k in ("ENTRYTYPE", "ID"):  # ya tratados
            continue
        v1 = e1.get(k, "")
        v2 = e2.get(k, "")
        if not v1 and not v2:
            continue

        lk = k.lower()
        # Campos de lista → unión
        if lk in LIST_FIELDS:
            merged[k] = smart_union_list([v1, v2], sep_candidates=LIST_FIELDS[lk])
            continue

        # doi / url → unión sin repetir
        if lk in {"doi", "url"}:
            merged[k] = smart_union_list([v1, v2], sep_candidates=(";", ",", " "))
            continue

        # abstract → concatenar sin repetir
        if lk == "abstract":
            if not v1: merged[k] = v2
            elif not v2: merged[k] = v1
            else:
                merged[k] = merge_scalar(v1, v2)
            continue

        # title → mantener el más informativo (pero no cambiamos clave de dedup)
        if lk == "title":
            merged[k] = merge_scalar(v1, v2)
            continue

        # Resto de campos escalares → merge sin perder información
        merged[k] = merge_scalar(v1, v2)

    return merged

# ---------------- Diagnóstico de incompletos (CSV) ----------------  # NOTE

def detectar_incompletos(entries: List[Dict]) -> List[Tuple[str, Dict[str, bool]]]:
    """
    Devuelve lista de tuplas (id, flags) donde flags marca falta de title/year/abstract.
    """
    res: List[Tuple[str, Dict[str, bool]]] = []
    for e in entries:
        id_ = str(e.get("ID", "noid"))
        title_ok = bool(normalized_title(e))
        year = (e.get("year") or e.get("Year") or e.get("YEAR") or "").strip()
        year_ok = bool(re.match(r"^\d{4}", year))  # simple heurística: 4 dígitos iniciales
        abstract_ok = bool((e.get("abstract") or "").strip())
        if not (title_ok and year_ok and abstract_ok):
            res.append((id_, {
                "missing_title": not title_ok,
                "missing_year": not year_ok,
                "missing_abstract": not abstract_ok
            }))
    return res

def exportar_incompletos_csv(items: List[Tuple[str, Dict[str, bool]]], ruta_csv: Path) -> None:
    with open(ruta_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "missing_title", "missing_year", "missing_abstract"])
        for id_, flags in items:
            writer.writerow([
                id_,
                int(flags["missing_title"]),
                int(flags["missing_year"]),
                int(flags["missing_abstract"]),
            ])

# ---------------- main (ejecutable) ----------------

def main():
    print(f"📥 Leyendo:\n - {ACM_BIB}\n - {SCIENCE_BIB}")
    acm_entries = cargar_bib(ACM_BIB)
    sd_entries = cargar_bib(SCIENCE_BIB)

    all_entries: List[Dict] = []
    all_entries.extend(acm_entries)
    all_entries.extend(sd_entries)

    if not all_entries:
        print("❌ No se encontraron entradas en los .bib de origen.")
        sys.exit(1)

    # Diagnóstico de incompletos  # NOTE
    incompletos = detectar_incompletos(all_entries)
    if incompletos:
        exportar_incompletos_csv(incompletos, INCOMPLETOS_CSV)
        print(f"📝 Registros incompletos listados en: {INCOMPLETOS_CSV}")
    else:
        try:
            if INCOMPLETOS_CSV.exists():
                INCOMPLETOS_CSV.unlink()
        except Exception:
            pass

    # Deduplicar por título normalizado
    by_title: Dict[str, Dict] = {}
    duplicados_descartados: List[Dict] = []  # NOTE
    for e in all_entries:
        tkey = normalized_title(e)
        if not tkey:
            tkey = f"__notitle__::{e.get('ID','noid')}"
        if tkey in by_title:
            duplicados_descartados.append(e)  # recolectar duplicados  # NOTE
            by_title[tkey] = merge_entries(by_title[tkey], e)
        else:
            by_title[tkey] = e

    # Reasignar ID seguro
    final_entries: List[Dict] = []
    for idx, (tkey, e) in enumerate(by_title.items(), start=1):
        if not e.get("ID"):
            base = re.sub(r"\s+", "_", tkey)[:40].strip("_")
            e["ID"] = f"{base or 'entry'}_{idx}"
        final_entries.append(e)

    print(f"🧮 Entradas originales: {len(all_entries)}")
    print(f"✅ Entradas tras deduplicar: {len(final_entries)}")
    print(f"♻️ Duplicados descartados (registrados): {len(duplicados_descartados)}")  # NOTE

    guardar_bib(final_entries, SALIDA_BIB)
    print(f"💾 Archivo unificado escrito en: {SALIDA_BIB}")

    # Guardar duplicados descartados  # NOTE
    if duplicados_descartados:
        guardar_bib(duplicados_descartados, DUPS_BIB)
        print(f"📄 Archivo con duplicados eliminados: {DUPS_BIB}")
    else:
        try:
            if DUPS_BIB.exists():
                DUPS_BIB.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    main()
