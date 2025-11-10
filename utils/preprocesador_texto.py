#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para preprocesamiento de texto académico.
"""

import re
import unicodedata
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Descargando recursos de NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class PreprocesadorTexto:
    """Clase para preprocesamiento de texto académico."""
    
    # Stopwords específicas del dominio
    STOPWORDS_DOMINIO = {
        'abstract', 'introduction', 'conclusion', 'paper', 'study',
        'research', 'results', 'findings', 'show', 'shows', 'shown',
        'propose', 'proposed', 'method', 'methods', 'using', 'used',
        'based', 'one', 'two', 'three', 'first', 'second', 'third',
        'however', 'thus', 'therefore', 'furthermore', 'moreover',
        'since', 'while', 'although', 'despite', 'hence', 'finally',
        'example', 'examples', 'well', 'may', 'also', 'fig', 'figure',
        'table', 'data', 'use', 'using', 'used', 'uses', 'et', 'al'
    }
    
    def __init__(self):
        """Inicializa el preprocesador con recursos necesarios."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(self.STOPWORDS_DOMINIO)
        
        # Expresiones regulares compiladas para eficiencia
        self.patron_numeros = re.compile(r'\d+')
        self.patron_puntuacion = re.compile(r'[^\w\s]')
        self.patron_espacios = re.compile(r'\s+')
        
    def _normalizar_texto(self, texto: str) -> str:
        """
        Normaliza el texto eliminando acentos y convirtiendo a minúsculas.
        
        Args:
            texto: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar acentos
        texto = unicodedata.normalize('NFKD', texto)
        texto = texto.encode('ASCII', 'ignore').decode('ASCII')
        
        return texto
    
    def _eliminar_caracteres_especiales(self, texto: str) -> str:
        """
        Elimina números, puntuación y caracteres especiales.
        
        Args:
            texto: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        # Eliminar números
        texto = self.patron_numeros.sub(' ', texto)
        
        # Eliminar puntuación
        texto = self.patron_puntuacion.sub(' ', texto)
        
        # Normalizar espacios
        texto = self.patron_espacios.sub(' ', texto)
        
        return texto.strip()
    
    def _lematizar_token(self, token: str) -> str:
        """
        Lematiza un token usando WordNet.
        
        Args:
            token: Token a lematizar
            
        Returns:
            Token lematizado
        """
        return self.lemmatizer.lemmatize(token)
    
    def _es_token_valido(self, token: str) -> bool:
        """
        Verifica si un token es válido según criterios específicos.
        
        Args:
            token: Token a verificar
            
        Returns:
            True si el token es válido, False en caso contrario
        """
        # Longitud mínima de 3 caracteres
        if len(token) < 3:
            return False
            
        # No debe ser stopword
        if token in self.stop_words:
            return False
            
        # Debe contener al menos una letra
        if not any(c.isalpha() for c in token):
            return False
            
        return True
    
    def _extraer_unigramas(self, texto: str) -> List[str]:
        """
        Extrae unigramas válidos del texto.
        
        Args:
            texto: Texto a procesar
            
        Returns:
            Lista de unigramas válidos
        """
        # Usar split() como alternativa a word_tokenize
        tokens = texto.split()
        return [
            self._lematizar_token(token)
            for token in tokens
            if self._es_token_valido(token)
        ]
    
    def _extraer_bigramas(self, texto: str) -> List[str]:
        """
        Extrae bigramas válidos del texto.
        
        Args:
            texto: Texto a procesar
            
        Returns:
            Lista de bigramas válidos
        """
        tokens = texto.split()
        bigramas = []
        
        for i in range(len(tokens) - 1):
            token1 = tokens[i]
            token2 = tokens[i + 1]
            
            # Verificar cada token individualmente
            if not (self._es_token_valido(token1) and self._es_token_valido(token2)):
                continue
                
            # Lematizar tokens
            token1 = self._lematizar_token(token1)
            token2 = self._lematizar_token(token2)
            
            # Unir tokens
            bigrama = f"{token1}_{token2}"
            bigramas.append(bigrama)
            
        return bigramas
    
    def procesar_texto(self, texto: str, incluir_bigramas: bool = False) -> List[str]:
        """
        Procesa el texto aplicando todas las transformaciones necesarias.
        
        Args:
            texto: Texto a procesar
            incluir_bigramas: Si True, incluye bigramas en el resultado
            
        Returns:
            Lista de tokens procesados
        """
        if not texto:
            return []
            
        # Normalización básica
        texto = self._normalizar_texto(texto)
        texto = self._eliminar_caracteres_especiales(texto)
        
        # Extraer tokens
        tokens = self._extraer_unigramas(texto)
        
        # Agregar bigramas si se solicitan
        if incluir_bigramas:
            tokens.extend(self._extraer_bigramas(texto))
            
        return tokens
    
    def obtener_stopwords(self) -> Set[str]:
        """
        Retorna el conjunto de stopwords utilizadas.
        
        Returns:
            Conjunto de stopwords
        """
        return self.stop_words.copy()

def main():
    """Función principal para pruebas del módulo."""
    # Texto de prueba académico
    texto = """
    This research paper presents an efficient algorithm for optimization
    of computational complexity in large-scale systems. Our methodology
    demonstrates significant performance improvements compared to
    traditional approaches. The experimental results show a 25% reduction
    in execution time and 30% lower computational cost.
    """
    
    preprocesador = PreprocesadorTexto()
    
    print("\nTexto original:")
    print(texto.strip())
    
    print("\nTokens procesados (solo unigramas):")
    tokens = preprocesador.procesar_texto(texto, incluir_bigramas=False)
    print(tokens)
    
    print("\nTokens procesados (con bigramas):")
    tokens_con_bigramas = preprocesador.procesar_texto(texto, incluir_bigramas=True)
    print(tokens_con_bigramas)
    
    print("\nStopwords específicas del dominio:")
    print(sorted(list(preprocesador.STOPWORDS_DOMINIO)))

if __name__ == "__main__":
    main()