#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para descargar recursos de NLTK necesarios para el proyecto.
"""

import nltk

def descargar_recursos():
    """Descarga todos los recursos necesarios de NLTK."""
    recursos = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger'
    ]
    
    for recurso in recursos:
        print(f"Descargando {recurso}...")
        nltk.download(recurso)

if __name__ == "__main__":
    print("Iniciando descarga de recursos NLTK...")
    descargar_recursos()
    print("Descarga completada.")