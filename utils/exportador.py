#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para exportación de resultados en diferentes formatos (CSV, JSON, PDF).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import csv
import pandas as pd

class Exportador:
    """Clase para exportar datos en diferentes formatos."""
    
    @staticmethod
    def a_csv(datos: Union[List[Dict], pd.DataFrame],
              ruta_salida: Union[str, Path],
              columnas: List[str] = None,
              encoding: str = 'utf-8') -> Path:
        """
        Exporta datos a CSV.
        
        Args:
            datos: Lista de diccionarios o DataFrame a exportar
            ruta_salida: Ruta donde guardar el CSV
            columnas: Lista de columnas a incluir (opcional)
            encoding: Codificación del archivo
            
        Returns:
            Path al archivo CSV generado
        """
        ruta_salida = Path(ruta_salida)
        
        # Convertir a DataFrame si es necesario
        if isinstance(datos, list):
            df = pd.DataFrame(datos)
        else:
            df = datos
            
        # Seleccionar columnas si se especifican
        if columnas:
            df = df[columnas]
            
        # Crear directorio si no existe
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar CSV
        df.to_csv(ruta_salida, index=False, encoding=encoding)
        return ruta_salida
    
    @staticmethod
    def a_json(datos: Union[Dict, List],
               ruta_salida: Union[str, Path],
               encoding: str = 'utf-8',
               indent: int = 2) -> Path:
        """
        Exporta datos a JSON.
        
        Args:
            datos: Diccionario o lista a exportar
            ruta_salida: Ruta donde guardar el JSON
            encoding: Codificación del archivo
            indent: Sangría para formato legible
            
        Returns:
            Path al archivo JSON generado
        """
        ruta_salida = Path(ruta_salida)
        
        # Crear directorio si no existe
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar JSON
        with open(ruta_salida, 'w', encoding=encoding) as f:
            json.dump(datos, f, ensure_ascii=False, indent=indent)
        
        return ruta_salida
    
    @staticmethod
    def df_a_pdf(df: pd.DataFrame,
                 ruta_salida: Union[str, Path],
                 titulo: str = "",
                 orientacion: str = 'portrait') -> Path:
        """
        Exporta un DataFrame a PDF.
        
        Args:
            df: DataFrame a exportar
            ruta_salida: Ruta donde guardar el PDF
            titulo: Título opcional para el documento
            orientacion: 'portrait' o 'landscape'
            
        Returns:
            Path al archivo PDF generado
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, landscape
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            raise ImportError("Se requiere reportlab para exportar a PDF")
            
        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        
        # Configurar documento
        if orientacion == 'landscape':
            pagesize = landscape(letter)
        else:
            pagesize = letter
            
        doc = SimpleDocTemplate(str(ruta_salida), pagesize=pagesize)
        
        # Crear elementos
        elementos = []
        styles = getSampleStyleSheet()
        
        # Agregar título si se especifica
        if titulo:
            elementos.append(Paragraph(titulo, styles['Heading1']))
        
        # Convertir DataFrame a tabla
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data)
        
        # Estilo de tabla
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        table.setStyle(style)
        elementos.append(table)
        
        # Generar PDF
        doc.build(elementos)
        return ruta_salida

def main():
    """Función principal para pruebas del módulo."""
    # Datos de prueba
    datos_dict = [
        {"id": 1, "nombre": "Test 1", "valor": 100},
        {"id": 2, "nombre": "Test 2", "valor": 200}
    ]
    
    # Crear directorio para pruebas
    dir_pruebas = Path("pruebas_exportacion")
    dir_pruebas.mkdir(exist_ok=True)
    
    # Probar exportación a CSV
    ruta_csv = dir_pruebas / "test.csv"
    Exportador.a_csv(datos_dict, ruta_csv)
    print(f"CSV creado en: {ruta_csv}")
    
    # Probar exportación a JSON
    ruta_json = dir_pruebas / "test.json"
    Exportador.a_json(datos_dict, ruta_json)
    print(f"JSON creado en: {ruta_json}")
    
    # Probar exportación a PDF
    df = pd.DataFrame(datos_dict)
    ruta_pdf = dir_pruebas / "test.pdf"
    try:
        Exportador.df_a_pdf(df, ruta_pdf, "Datos de Prueba")
        print(f"PDF creado en: {ruta_pdf}")
    except ImportError as e:
        print(f"No se pudo crear PDF: {e}")

if __name__ == "__main__":
    main()