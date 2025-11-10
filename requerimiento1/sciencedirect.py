# requerimiento1/sciencedirect.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, glob, urllib.parse, random
from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

load_dotenv()


class ScienceDirectDescarga:
    def __init__(self, query_text="generative artificial intelligence", per_page=100):
        """
        Sin límite de páginas: navega hasta que no haya más resultados.
        """
        self.download_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "descargas", "ScienceDirect")
        )
        os.makedirs(self.download_dir, exist_ok=True)

        env_per_page = os.getenv("SD_PER_PAGE")
        self.per_page = int(env_per_page) if (env_per_page and env_per_page.isdigit()) else int(per_page or 100)
        self.per_page = min(max(self.per_page, 25), 200)

        qt = (query_text or "").strip()
        if not (qt.startswith('"') and qt.endswith('"')):
            qt = f'"{qt}"'
        self.query_text = qt

        self.driver = self._configurar_chrome()

        self._bib_previos = {f for f in os.listdir(self.download_dir) if f.lower().endswith(".bib")}
        self._bib_ignorar = {"acmcompleto.bib", "sciencedirectcompleto.bib", "resultado_unificado.bib"}

    # -------------------- helpers --------------------
    def _wait_click(self, by, value, timeout=25, desc="elemento"):
        try:
            return WebDriverWait(self.driver, timeout, poll_frequency=0.5).until(
                EC.element_to_be_clickable((by, value))
            )
        except TimeoutException as e:
            print(f"⏳ Timeout esperando {desc}: {by} -> {value} (url: {self.driver.current_url})")
            raise e

    def _wait_presence(self, by, value, timeout=25, desc="elemento"):
        try:
            return WebDriverWait(self.driver, timeout, poll_frequency=0.5).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException as e:
            print(f"⏳ Timeout esperando presencia de {desc}: {by} -> {value} (url: {self.driver.current_url})")
            raise e

    def _js_click(self, el):
        self.driver.execute_script("arguments[0].click();", el)

    def _type(self, el, text):
        """Escritura robusta: ActionChains y fallback JS."""
        try:
            el.click()
            try:
                el.clear()
            except Exception:
                pass
            ActionChains(self.driver).move_to_element(el).click(el).send_keys(text).perform()
            if (el.get_attribute("value") or "").strip() != text:
                self.driver.execute_script(
                    "arguments[0].value=arguments[1];"
                    "arguments[0].dispatchEvent(new Event('input',{bubbles:true}));"
                    "arguments[0].dispatchEvent(new Event('change',{bubbles:true}));",
                    el,
                    text,
                )
        except Exception:
            self.driver.execute_script("arguments[0].value=arguments[1];", el, text)

    def _url_contains(self, substrs, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            url = (self.driver.current_url or "").lower()
            if any(s in url for s in substrs):
                return True
            time.sleep(0.25)
        return False

    def _esperar_nuevo_bib(self, timeout=60):
        inicio = time.time()
        antes = set(glob.glob(os.path.join(self.download_dir, "*.bib")))
        while time.time() - inicio < timeout:
            time.sleep(1)
            if set(glob.glob(os.path.join(self.download_dir, "*.bib"))) - antes:
                return True
        return False

    def _sleep_human(self, base=2.0, jitter=1.0):
        time.sleep(base + random.random() * jitter)

    # -------------------- URL helpers (paginación robusta) --------------------
    def _parse_offset(self, url: str) -> int:
        """Extrae offset=? de la URL; si no existe, intenta deducirlo por 'Page X of Y'."""
        try:
            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)
            if "offset" in qs and qs["offset"]:
                return int(qs["offset"][0])
        except Exception:
            pass
        # Deducción por pager en el DOM
        try:
            txt = self.driver.find_element(By.CSS_SELECTOR, "ol#srp-pagination").text
            parts = txt.split()  # "Page 12 of 64"
            if "Page" in parts and "of" in parts:
                idx = parts.index("Page")
                page_num = int(parts[idx + 1])
                return (page_num - 1) * self.per_page
        except Exception:
            pass
        return 0

    def _build_url_with_offset(self, offset: int) -> str:
        q = urllib.parse.quote_plus(self.query_text or "")
        # Conservamos proxy y show
        return f"https://www-sciencedirect-com.crai.referencistas.com/search?qs={q}&show={self.per_page}&offset={offset}"

    # -------------------- config (Selenium "normal") --------------------
    def _configurar_chrome(self):
        chrome_bin = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
        chromedriver_path = os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

        opts = Options()
        opts.binary_location = chrome_bin

        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.automatic_downloads": 1,
        }
        opts.add_experimental_option("prefs", prefs)

        # Flags recomendadas para contenedores/headless
        if os.getenv("SD_HEADLESS", "1") == "1":
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--disable-popup-blocking")
        opts.add_argument("--ignore-certificate-errors")

        service = Service(executable_path=chromedriver_path)
        drv = webdriver.Chrome(service=service, options=opts)

        # Intento de habilitar descargas en headless (algunos builds lo requieren)
        try:
            drv.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": self.download_dir},
            )
        except Exception:
            pass

        return drv

    # -------------------- navegación/login --------------------
    def _dismiss_banners(self):
        for by, sel in [
            (By.ID, "onetrust-accept-btn-handler"),
            (By.CSS_SELECTOR, "button[aria-label='accept cookies']"),
            (By.CSS_SELECTOR, "button[aria-label='Accept all']"),
        ]:
            try:
                self._js_click(WebDriverWait(self.driver, 3).until(EC.element_to_be_clickable((by, sel))))
                break
            except Exception:
                pass

    def _click_google_crai(self):
        """Clic robusto en 'Iniciar sesión con Google' (DOM + iframes)."""
        try:
            candidatos = [
                (By.XPATH, "//button[contains(.,'Iniciar sesión con Google')]"),
                (By.XPATH, "//a[contains(.,'Iniciar sesión con Google')]"),
                (By.XPATH, "//button[contains(.,'Sign in with Google')]"),
                (By.XPATH, "//a[contains(.,'Sign in with Google')]"),
                (By.CSS_SELECTOR, "a[href*='google']"),
                (By.CSS_SELECTOR, "button.btn.btn-success"),
                (By.CSS_SELECTOR, "a.btn.btn-success"),
                (By.XPATH, "//*[contains(.,'Google') and (self::a or self::button)]"),
            ]
            # DOM principal
            btn = None
            for by, sel in candidatos:
                try:
                    btn = WebDriverWait(self.driver, 6).until(EC.element_to_be_clickable((by, sel)))
                    break
                except Exception:
                    continue
            # iframes
            if not btn:
                for fr in self.driver.find_elements(By.TAG_NAME, "iframe"):
                    try:
                        self.driver.switch_to.default_content()
                        self.driver.switch_to.frame(fr)
                        for by, sel in candidatos:
                            try:
                                btn = WebDriverWait(self.driver, 3).until(EC.element_to_be_clickable((by, sel)))
                                if btn:
                                    break
                            except Exception:
                                continue
                        if btn:
                            break
                    except Exception:
                        continue
                self.driver.switch_to.default_content()

            if not btn:
                print("❌ No se encontró el botón 'Iniciar sesión con Google'.")
                return False

            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(1)
            try:
                btn.click()
            except Exception:
                self._js_click(btn)
            print("✅ Clic en 'Iniciar sesión con Google'.")
            return True
        except Exception as e:
            print(f"❌ Error _click_google_crai: {e}")
            return False

    def acceso_directo_sciencedirect(self):
        q = urllib.parse.quote_plus(self.query_text or "")
        url = f"https://www-sciencedirect-com.crai.referencistas.com/search?qs={q}&show={self.per_page}"
        print(f"🌐 Accediendo: {url}")
        self.driver.get(url)
        time.sleep(4)

        title = (self.driver.title or "").lower()
        if "biblioteca" in title or "crai" in (self.driver.page_source or "").lower():
            if self._click_google_crai():
                time.sleep(2)

        if any(s in (self.driver.current_url or "").lower() for s in ["login", "signin", "accounts.google.com", "idp", "saml"]):
            self.login_google_automatico()

        self._dismiss_banners()

    def login_google_automatico(self):
        email, pswd = os.getenv("EMAIL"), os.getenv("PSWD")
        if not email or not pswd:
            print("⚠ EMAIL/PSWD no definidos, continuaré sin login.")
            return False

        if not self._url_contains(["accounts.google.com", "intelproxy.com", "idp", "saml"], timeout=20):
            return False

        if "accounts.google.com" in (self.driver.current_url or "").lower():
            try:
                # Usar otra cuenta (si aparece)
                try:
                    btn = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "//div[@role='button'][.//div[contains(text(),'Usar otra cuenta') or contains(text(),'Use another account')]]",
                            )
                        )
                    )
                    btn.click()
                    time.sleep(0.8)
                except Exception:
                    pass

                # email
                email_box = None
                for loc in [(By.ID, "identifierId"), (By.CSS_SELECTOR, "input[type='email']"), (By.NAME, "identifier")]:
                    try:
                        email_box = WebDriverWait(self.driver, 8).until(EC.element_to_be_clickable(loc))
                        break
                    except Exception:
                        continue
                if not email_box:
                    return False
                self._type(email_box, email)

                for loc in [
                    (By.ID, "identifierNext"),
                    (
                        By.XPATH,
                        "//button[@type='button' or @type='submit'][.//span[text()='Siguiente' or text()='Next']]",
                    ),
                ]:
                    try:
                        WebDriverWait(self.driver, 6).until(EC.element_to_be_clickable(loc)).click()
                        break
                    except Exception:
                        continue
                time.sleep(2)

                # password (si lo pide)
                if "accounts.google.com" in (self.driver.current_url or "").lower():
                    try:
                        pwd = WebDriverWait(self.driver, 8).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
                        )
                        self._type(pwd, pswd)
                        for loc in [
                            (By.ID, "passwordNext"),
                            (
                                By.XPATH,
                                "//button[@type='button' or @type='submit'][.//span[text()='Siguiente' or text()='Next']]",
                            ),
                        ]:
                            try:
                                WebDriverWait(self.driver, 6).until(EC.element_to_be_clickable(loc)).click()
                                break
                            except Exception:
                                continue
                    except Exception:
                        pass
            except Exception:
                return False

        if self._url_contains(["intelproxy.com", "idp", "saml"], timeout=15):
            self._finish_after_sso(email, pswd)

        ok = self._url_contains(["sciencedirect.com", "referencistas.com"], timeout=30)
        if ok:
            print("✅ Login completado (ScienceDirect).")
        else:
            print(f"⚠ Login inconcluso. URL: {self.driver.current_url}")
        return ok

    def _finish_after_sso(self, email, pswd):
        try:
            for loc in [
                (By.CSS_SELECTOR, "input[type='email']"),
                (By.NAME, "email"),
                (By.NAME, "username"),
                (By.ID, "username"),
            ]:
                try:
                    self._type(WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable(loc)), email)
                    break
                except Exception:
                    pass
            for loc in [
                (By.CSS_SELECTOR, "input[type='password']"),
                (By.NAME, "password"),
                (By.ID, "password"),
            ]:
                try:
                    self._type(WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable(loc)), pswd)
                    break
                except Exception:
                    pass
            for loc in [
                (By.CSS_SELECTOR, "button[type='submit']"),
                (
                    By.XPATH,
                    "//button[contains(.,'Ingresar') or contains(.,'Acceder') or contains(.,'Login') or contains(.,'Sign in')]",
                ),
                (By.XPATH, "//input[@type='submit']"),
            ]:
                try:
                    WebDriverWait(self.driver, 6).until(EC.element_to_be_clickable(loc)).click()
                    break
                except Exception:
                    pass
        except Exception:
            pass

    # -------------------- scraping --------------------
    def descargar_pagina_actual(self):
        """Selecciona todo y exporta BibTeX de la página actual."""
        for intento in range(2):
            try:
                # asegurar que la paginación esté visible
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self._sleep_human(1.0, 0.8)

                label = self._wait_click(By.CSS_SELECTOR, "label[for='select-all-results']", 25, "select_all")
                cb = self.driver.find_element(By.ID, "select-all-results")
                if not cb.is_selected():
                    label.click()
                    self._sleep_human(0.6, 0.5)

                # abrir modal export
                self._wait_click(
                    By.XPATH, '//*[@id="srp-toolbar"]/div[1]/span/span[1]/span[2]/div[2]', 25, "export_toolbar"
                ).click()
                self._sleep_human(1.0, 0.5)

                # botón BibTeX
                self._wait_click(
                    By.CSS_SELECTOR, "button[data-aa-button='srp-export-multi-bibtex']", 25, "export_bibtex"
                ).click()

                # esperar el .bib
                self._esperar_nuevo_bib(60)

                # cerrar modal si siguiera
                try:
                    self._js_click(
                        WebDriverWait(self.driver, 4).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Close'], button.close"))
                        )
                    )
                except Exception:
                    pass
                print("✅ Exportación BibTeX OK en esta página.")
                return True
            except (TimeoutException, StaleElementReferenceException):
                print(f"🔁 Reintentando exportación...")
                try:
                    self.driver.execute_script("location.reload()")
                    self._sleep_human(4, 1.5)
                except Exception:
                    pass
        print("❌ Falló la exportación en esta página.")
        return False

    def _try_click_next(self) -> bool:
        """Intenta hacer clic en Next. Devuelve True si cree que navegó."""
        try:
            next_btn = self._wait_presence(By.CSS_SELECTOR, "a[data-aa-name='srp-next-page']", 10, "next_btn")
            disabled = (next_btn.get_attribute("aria-disabled") or "").lower() == "true"
            classes = next_btn.get_attribute("class") or ""
            if disabled or "is-disabled" in classes:
                return False
            btn = self._wait_click(By.CSS_SELECTOR, 'a[data-aa-name="srp-next-page"]', 15, "next_page")
            self._js_click(btn)
            return True
        except Exception:
            return False

    def siguiente_pagina(self) -> bool:
        """
        Paginación híbrida:
          1) Clic en Next.
          2) Verifica cambio de offset/página.
          3) Si no cambió, fuerza navegación con &offset=...
        """
        # offset/página actual
        current_url = self.driver.current_url
        before_offset = self._parse_offset(current_url)

        # intento 1: clic
        clicked = self._try_click_next()
        self._sleep_human(2.2, 1.2)

        changed = self._parse_offset(self.driver.current_url) > before_offset
        if not changed:
            # intento 2: forzar URL con offset
            forced_offset = before_offset + self.per_page
            force_url = self._build_url_with_offset(forced_offset)
            print(f"↪️ Forzando navegación a offset={forced_offset}")
            self.driver.get(force_url)
            self._sleep_human(2.5, 1.5)
            changed = self._parse_offset(self.driver.current_url) >= forced_offset

        if not changed:
            # intento 3: recargar y volver a forzar
            try:
                self.driver.execute_script("location.reload()")
                self._sleep_human(2.5, 1.2)
            except Exception:
                pass
            forced_offset = before_offset + self.per_page
            self.driver.get(self._build_url_with_offset(forced_offset))
            self._sleep_human(2.5, 1.5)
            changed = self._parse_offset(self.driver.current_url) >= forced_offset

        return changed

    def unir_archivos_bibtex(self):
        """Une SOLO los .bib nuevos y borra únicamente esos."""
        time.sleep(3)
        actuales = [f for f in os.listdir(self.download_dir) if f.lower().endswith(".bib")]
        nuevos = [f for f in actuales if f not in self._bib_previos and f.lower() not in self._bib_ignorar]
        if not nuevos:
            print("ℹ No se encontraron .bib nuevos para unir.")
            return None

        out = os.path.join(self.download_dir, "sciencedirectCompleto.bib")
        with open(out, "w", encoding="utf-8") as w:
            for f in nuevos:
                try:
                    with open(os.path.join(self.download_dir, f), "r", encoding="utf-8") as r:
                        c = r.read().strip()
                        if c:
                            w.write(c + "\n\n")
                    print(f"✅ Procesado: {f}")
                except Exception as e:
                    print(f"⚠ Error leyendo {f}: {e}")

        for f in nuevos:
            try:
                os.remove(os.path.join(self.download_dir, f))
            except Exception:
                pass
        print(f"📁 BibTeX unificado: {out}")
        return out

    # -------------------- interfaz pública --------------------
    def abrir_base_datos(self):
        print("🔗 Método abrir_base_datos() - Ejecutando descarga...")
        return self.ejecutar_descarga()

    def ejecutar_descarga(self):
        """
        Pipeline completo SIN LÍMITES:
        - Login/CRAI si aplica.
        - Exporta BibTeX por página.
        - Página siguiente hasta agotar resultados (clic o URL forzada).
        """
        try:
            self.acceso_directo_sciencedirect()
            if not self._url_contains(["sciencedirect.com", "referencistas.com"], timeout=15):
                print("⚠ No estamos en ScienceDirect; podría fallar la paginación.")

            pagina = 1
            while True:
                print(f"📄 Página {pagina} (offset actual: {self._parse_offset(self.driver.current_url)})")
                self.descargar_pagina_actual()

                if not self.siguiente_pagina():
                    print("🚩 Fin de resultados (Next no disponible o offset no avanza).")
                    break

                pagina += 1
                # Pausa “humana” para evitar rate-limits/detecciones
                self._sleep_human(1.8, 1.4)

            unido = self.unir_archivos_bibtex()
            ok = bool(unido)
            if ok:
                print("🎉 Descarga completa (todas las páginas disponibles).")
            else:
                print("❌ No se pudo crear el archivo unificado.")
            return ok
        except Exception as e:
            print(f"💥 Error crítico: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cerrar(self):
        try:
            self.driver.quit()
            print("🔚 Navegador cerrado")
        except Exception:
            pass
        finally:
            self.driver = None


# Compatibilidad
def download_sciense_articles():
    d = ScienceDirectDescarga()
    try:
        return d.ejecutar_descarga()
    finally:
        d.cerrar()
