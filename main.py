# main.py
from requerimiento1.acm_descarga import ACMDescarga
from requerimiento1.sciencedirect import ScienceDirectDescarga  # <-- corrección de nombre

# Import del unificador (principal) con fallback si el archivo se llama distinto
try:
    from requerimiento1.unir_bib_deduplicado import main as unir_bib
except ImportError:
    try:
        # Fallback alternativo si el archivo se llamara unir_bib.py
        from requerimiento1.unir_bib_deduplicado import main as unir_bib
    except ImportError:
        raise

import traceback
import os


def run_sciencedirect(query: str = "generative artificial intelligence"):
    print("=== [1/3] Iniciando ScienceDirect ===")
    sd = ScienceDirectDescarga(query)
    try:
        sd.ejecutar_descarga()
    finally:
        try:
            sd.cerrar()
        except Exception:
            pass
    print("=== [1/3] ScienceDirect finalizado ===\n")


def run_acm():
    print("=== [2/3] Iniciando ACM ===")
    bot = ACMDescarga()
    try:
        bot.abrir_base_datos()
    finally:
        try:
            bot.cerrar()
        except Exception:
            pass
    print("=== [2/3] ACM finalizado ===\n")


def run_unificador():
    print("=== [3/3] Unificando BibTeX (ACM + ScienceDirect) ===")
    try:
        # escribe 'resultado_unificado.bib' en la carpeta de descargas configurada
        unir_bib()
        print("=== [3/3] Unificación completada ===\n")
    except Exception:
        print("❌ Error durante la unificación:")
        traceback.print_exc()


if __name__ == "__main__":
    # Crear las carpetas necesarias antes de comenzar
    base_dir = os.path.join(os.path.dirname(__file__), "requerimiento1", "descargas")
    acm_dir = os.path.join(base_dir, "acm")
    sd_dir = os.path.join(base_dir, "ScienceDirect")

    for dir_path in [base_dir, acm_dir, sd_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 1) Ejecutar ScienceDirect primero (ajusta el query si quieres)
    # run_sciencedirect("generative artificial intelligence")

    # 2) Ejecutar ACM después
    # run_acm()

    # 3) Unificar ambos .bib en un solo archivo
    run_unificador()

    print("🎉 Flujo completo finalizado.")
