#!/bin/bash

# Definir las bibliotecas necesarias
libraries=("numpy" "pandas" "seaborn" "matplotlib" "unidecode" "spacy" "-U pip setuptools wheel" "pyspellchecker" "sklearn" "indexer" "reportlab" "datetime")

# Verificar si Python está instalado
if command -v python3 &>/dev/null; then
    for library in "${libraries[@]}"; do
        # Verificar si la biblioteca ya está instalada
        if python3 -c "import $library" 2>/dev/null; then
            echo "$library ya está instalada."
        else
            echo "Instalando $library..."
            pip install $library
        fi
    done
else
    echo "Python 3 no está instalado en este sistema."
fi

# Install spacy spanish and english en_core_web_sm and es_core_news_sm
if python3 -c "import spacy" 2>/dev/null; then
    echo "Instalando modelos de spacy..."
    python3 -m spacy download en_core_web_sm
    python3 -m spacy download es_core_news_sm
else
    echo "spacy no está instalado en este sistema."
fi