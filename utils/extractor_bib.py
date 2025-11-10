#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para cargar y parsear archivos BibTeX del proyecto bibliométrico.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import bibtexparser
from bibtexparser.bibdatabase import BibDatabase

class ExtratorBibTeX:
    """Clase para extraer y procesar información de archivos BibTeX."""
    
    def __init__(self, ruta_bib: Union[str, Path]):
        """
        Inicializa el extractor BibTeX.
        
        Args:
            ruta_bib: Ruta al archivo BibTeX a procesar
        """
        self.ruta_bib = Path(ruta_bib)
        self._entries: List[Dict] = []
        self._load_bib()
    
    def _load_bib(self) -> None:
        """Carga el archivo BibTeX usando bibtexparser."""
        if not self.ruta_bib.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.ruta_bib}")
            
        try:
            with open(self.ruta_bib, 'r', encoding='utf-8') as bibfile:
                db = bibtexparser.load(bibfile)
                self._entries = db.entries
        except Exception as e:
            raise RuntimeError(f"Error al cargar el archivo BibTeX: {e}")
    
    def get_abstracts(self) -> List[Dict[str, str]]:
        """
        Obtiene los abstracts de todas las entradas.
        
        Returns:
            Lista de diccionarios con ID y abstract de cada entrada
        """
        return [
            {"id": entry.get("ID", ""), "abstract": entry.get("abstract", "")}
            for entry in self._entries
            if entry.get("abstract")
        ]
    
    def get_titulos(self) -> List[Dict[str, str]]:
        """
        Obtiene los títulos de todas las entradas.
        
        Returns:
            Lista de diccionarios con ID y título de cada entrada
        """
        return [
            {"id": entry.get("ID", ""), "titulo": entry.get("title", "")}
            for entry in self._entries
            if entry.get("title")
        ]
    
    def get_autores(self) -> List[Dict[str, str]]:
        """
        Obtiene los autores de todas las entradas.
        
        Returns:
            Lista de diccionarios con ID y autores de cada entrada
        """
        return [
            {"id": entry.get("ID", ""), "autores": entry.get("author", "")}
            for entry in self._entries
            if entry.get("author")
        ]
    
    def get_keywords(self) -> List[Dict[str, str]]:
        """
        Obtiene las palabras clave de todas las entradas.
        
        Returns:
            Lista de diccionarios con ID y keywords de cada entrada
        """
        return [
            {"id": entry.get("ID", ""), "keywords": entry.get("keywords", "")}
            for entry in self._entries
            if entry.get("keywords")
        ]
    
    def get_years(self) -> List[Dict[str, str]]:
        """
        Obtiene los años de publicación de todas las entradas.
        
        Returns:
            Lista de diccionarios con ID y año de cada entrada
        """
        return [
            {"id": entry.get("ID", ""), "year": entry.get("year", "")}
            for entry in self._entries
            if entry.get("year")
        ]
    
    def get_all_fields(self, entry_id: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Obtiene todos los campos de una entrada específica o de todas las entradas.
        
        Args:
            entry_id: ID de la entrada específica (opcional)
            
        Returns:
            Diccionario con todos los campos de la entrada especificada,
            o lista de diccionarios con todas las entradas si no se especifica ID
        """
        if entry_id:
            for entry in self._entries:
                if entry.get("ID") == entry_id:
                    return dict(entry)
            raise ValueError(f"No se encontró entrada con ID: {entry_id}")
        return self._entries
    
    @property
    def total_entries(self) -> int:
        """Retorna el número total de entradas en el archivo BibTeX."""
        return len(self._entries)

def main():
    """Función principal para pruebas del módulo."""
    # Ruta al archivo BibTeX unificado
    ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
    
    try:
        extractor = ExtratorBibTeX(ruta_bib)
        print(f"Total de entradas: {extractor.total_entries}")
        
        # Ejemplo de extracción de abstracts
        abstracts = extractor.get_abstracts()
        print(f"\nTotal de abstracts: {len(abstracts)}")
        if abstracts:
            print("\nPrimer abstract:")
            print(abstracts[0])
            
        # Ejemplo de extracción de títulos
        titulos = extractor.get_titulos()
        print(f"\nTotal de títulos: {len(titulos)}")
        if titulos:
            print("\nPrimer título:")
            print(titulos[0])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()