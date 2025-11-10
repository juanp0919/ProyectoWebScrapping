#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
driver.py
---------
Configuración estándar del navegador Chrome/Chromium para scraping automatizado.
Compatible con entornos Docker (Render, local, etc.) y ejecución headless.
"""

import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def build_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Crea y devuelve un driver de Chrome listo para scraping.

    Args:
        headless (bool): Si True, corre en modo invisible (sin interfaz gráfica).

    Returns:
        selenium.webdriver.Chrome: driver configurado.
    """
    chrome_bin = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
    chromedriver_path = os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

    chrome_options = Options()
    chrome_options.binary_location = chrome_bin

    if headless:
        chrome_options.add_argument("--headless=new")

    # Opciones recomendadas para contenedores
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-dev-tools")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    # Servicio de ChromeDriver
    service = Service(executable_path=chromedriver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


if __name__ == "__main__":
    # Pequeña prueba local
    print("Iniciando prueba de driver...")
    try:
        driver = build_driver()
        driver.get("https://www.google.com")
        print("Título de página:", driver.title)
        driver.quit()
    except Exception as e:
        print("Error inicializando el driver:", e)
