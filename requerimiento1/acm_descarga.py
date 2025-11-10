#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class ACMDescarga:
    """
    Descarga resultados de ACM Digital Library (BibTeX) para la query
    "generative artificial intelligence" a través del portal de la UQ.
    Concatena los .bib NUEVOS de esta ejecución en acmCompleto.bib.
    """

    def __init__(self):
        # Carpeta de descargas (relativa a este archivo)
        self.download_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "descargas", "acm")
        )
        os.makedirs(self.download_dir, exist_ok=True)

        # ==== Selenium Chrome/Chromium (SIN undetected_chromedriver) ====
        chrome_bin = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
        chromedriver_path = os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

        options = Options()
        options.binary_location = chrome_bin

        # Preferencias de descarga
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_settings.popups": 0,
        }
        options.add_experimental_option("prefs", prefs)

        # Flags recomendadas para contenedores/headless
        if os.getenv("ACM_HEADLESS", "1") == "1":
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--ignore-certificate-errors")

        service = Service(executable_path=chromedriver_path)
        self.driver = webdriver.Chrome(service=service, options=options)

        # Snapshot de .bib existentes antes de iniciar
        self._bib_previos = set(
            f for f in os.listdir(self.download_dir) if f.lower().endswith(".bib")
        )

    # -------------------- Helpers de espera/click --------------------

    def _wait_click(self, by, value, timeout=30, desc="elemento"):
        """Espera a que un elemento sea clickable y lo devuelve."""
        try:
            w = WebDriverWait(self.driver, timeout, poll_frequency=0.5)
            return w.until(EC.element_to_be_clickable((by, value)))
        except TimeoutException as e:
            print(f"⏳ Timeout esperando click en {desc}: {by} -> {value}")
            raise e

    def _wait_presence(self, by, value, timeout=30, desc="elemento"):
        """Espera a que un elemento esté presente en el DOM y lo devuelve."""
        try:
            w = WebDriverWait(self.driver, timeout, poll_frequency=0.5)
            return w.until(EC.presence_of_element_located((by, value)))
        except TimeoutException as e:
            print(f"⏳ Timeout esperando presencia de {desc}: {by} -> {value}")
            raise e

    def _js_click(self, element):
        """Click con JavaScript para elementos que no responden al click normal."""
        self.driver.execute_script("arguments[0].click();", element)

    def _esperar_nuevo_bib(self, timeout=60):
        """Espera a que aparezca un nuevo .bib en la carpeta de descargas."""
        inicio = time.time()
        antes = set(glob.glob(os.path.join(self.download_dir, "*.bib")))
        while time.time() - inicio < timeout:
            time.sleep(1)
            despues = set(glob.glob(os.path.join(self.download_dir, "*.bib")))
            nuevos = despues - antes
            if nuevos:
                return True
        return False

    # -------------------- Flujo principal --------------------

    def abrir_base_datos(self):
        """Navega via portal UQ → ACM, realiza la búsqueda y exporta en lotes."""
        portal = "https://library.uniquindio.edu.co/databases"
        self.driver.get(portal)

        wait = WebDriverWait(self.driver, 20)
        wait_long = WebDriverWait(self.driver, 45)

        # Oculta overlay de carga si existe
        try:
            wait.until(EC.invisibility_of_element_located((By.CLASS_NAME, "onload-background")))
        except Exception:
            pass

        enlace = self._wait_click(By.LINK_TEXT, "BASES DATOS x FACULTAD", 25, "link bases por facultad")
        self._js_click(enlace)

        try:
            wait.until(EC.invisibility_of_element_located((By.CLASS_NAME, "onload-background")))
        except Exception:
            pass

        fac = self._wait_click(
            By.XPATH, "//div[@data-content-listing-item='fac-ingenier-a']",
            25, "facultad ingeniería"
        )
        self._js_click(fac)

        acm_link = self._wait_click(
            By.XPATH, "//a[@href='https://dl.acm.org/']",
            25, "link ACM"
        )
        self._js_click(acm_link)

        # Cambiar a nueva pestaña
        self.driver.switch_to.window(self.driver.window_handles[-1])

        # Búsqueda
        search_box = self._wait_presence(By.NAME, "AllField", 25, "caja de búsqueda")
        search_box.clear()
        search_box.send_keys('"generative artificial intelligence"')

        search_btn = self._wait_click(By.CSS_SELECTOR, "button.quick-search__button", 25, "botón buscar")
        self._js_click(search_btn)

        wait_long.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.search__item.issue-item-container"))
        )

        # Cambiar a 50 resultados por página
        try:
            link_50 = self._wait_click(
                By.XPATH, "//div[@class='per-page separator-end']//a[contains(@href,'pageSize=50')]",
                20, "link 50 por página"
            )
            self._js_click(link_50)
            time.sleep(4)
        except Exception:
            print("⚠ No se pudo cambiar a 50 por página; continúo con el valor por defecto.")

        # Procesar todas las páginas
        pagina = 0
        while True:
            print(f"📄 Procesando página {pagina + 1}...")
            wait_long.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.search__item.issue-item-container"))
            )

            # Seleccionar todos
            try:
                checkbox = self._wait_click(By.CSS_SELECTOR, "input[name='markall']", 20, "checkbox marcar todo")
                self._js_click(checkbox)
                time.sleep(1)
            except Exception as e:
                print(f"⚠ No se pudo marcar 'Select all': {e}")

            # Exportar citaciones
            export_btn = self._wait_click(By.CSS_SELECTOR, "a.export-citation", 25, "export citation")
            self._js_click(export_btn)
            time.sleep(2)

            download_btn = self._wait_click(By.CSS_SELECTOR, "a.download__btn", 25, "download bibtex")
            self._js_click(download_btn)
            print(f"✅ Descarga iniciada en página {pagina + 1}")

            if not self._esperar_nuevo_bib(timeout=60):
                print("⚠ No se detectó un nuevo .bib a tiempo; continúo igualmente.")

            # Cerrar modal
            try:
                close_modal = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((
                        By.CSS_SELECTOR,
                        "button[title='Close'], button.close, button[data-dismiss='modal']"
                    ))
                )
                self._js_click(close_modal)
                time.sleep(1)
            except Exception:
                pass

            # Siguiente página
            try:
                next_btn = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a.pagination__btn--next"))
                )
                disabled = next_btn.get_attribute("aria-disabled")
                if (disabled and disabled.lower() == "true") or "is-disabled" in (next_btn.get_attribute("class") or ""):
                    print("🚀 Fin de resultados (Next deshabilitado).")
                    break
                self._js_click(next_btn)
                pagina += 1
                time.sleep(4)
            except Exception:
                print("🚀 Fin de resultados (no se encontró Next).")
                break

        self.unir_archivos()

    # -------------------- Unión de archivos --------------------

    def unir_archivos(self):
        """Une SOLO los archivos .bib descargados durante ESTA ejecución."""
        bib_actuales = [f for f in os.listdir(self.download_dir) if f.lower().endswith(".bib")]
        bib_nuevos = [f for f in bib_actuales if f not in self._bib_previos]

        if not bib_nuevos:
            print("ℹ No se detectaron .bib nuevos de ACM en esta ejecución.")
            return

        output_file = os.path.join(self.download_dir, "acmCompleto.bib")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for fname in bib_nuevos:
                ruta = os.path.join(self.download_dir, fname)
                try:
                    with open(ruta, "r", encoding="utf-8") as infile:
                        content = infile.read().strip()
                        if content:
                            outfile.write(content + "\n\n")
                    print(f"✅ Agregado: {fname}")
                except Exception as e:
                    print(f"❌ Error leyendo {fname}: {e}")

        print(f"📚 Archivos de ACM unidos en: {output_file}")

        # Limpiar SOLO los nuevos
        for f in bib_nuevos:
            try:
                os.remove(os.path.join(self.download_dir, f))
            except Exception as e:
                print(f"⚠ No se pudo eliminar {f}: {e}")

        print("🧹 Limpieza completada.")

    def cerrar(self):
        """Cierra el navegador de forma segura."""
        try:
            self.driver.quit()
        except Exception:
            pass
