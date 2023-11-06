'''
This script is used to analyze the results of the surveys. It returns a document report
where the most common words are shown for each course and the most common topics are shown
for all the courses.

Author : Daniel Malváez
Version : 1.0.0
Date : 2023-10-14

Catching up the notebook:
    1. Install the required libraries:
        $ ./intall-libraries.sh
    2. Run the script:
        $ python3 Analisis.py
    3. The script will generate PNG images with the results and a PDF report.
'''

# ------------------------------------
# Important libraries for the analysis
# ------------------------------------
# Data Manipulation & Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Collections
from collections import Counter

# System
import os

# NLP preprocessing
import spacy
from spellchecker import SpellChecker
from unidecode import unidecode

# Term Frequency - Inverse Document Frequency and TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Report Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle, TA_JUSTIFY

# Regular Expressions
import re

# DateTime
import datetime


sns.set_style('darkgrid')

# Defining important functions
def remove_stop_words(text, doc):
    '''
    This function receives a text and a spaCy document object, and returns the text
    with all the tokens in the document that are not stop words, punctuation, spaces, numbers, URLs or emails.
    
    Parameters
    ------------
    text (str): The original text.
    doc (spacy.tokens.doc.Doc): The spaCy document object.
    
    Returns
    ------------
    str: A string with all the tokens in the document that are not stop words, punctuation, spaces, numbers, URLs or emails.
    '''
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num and not token.like_url and not token.like_email]
    return ' '.join(tokens)

def plot_top_words(model, feature_names, n_top_words, title):
        '''
        Plots the top words for each topic in a given model.

        Parameters
        ------------
        - model: the model to extract the top words from.
        - feature_names: the feature names of the model.
        - n_top_words: the number of top words to plot.
        - title: the title of the plot.

        Returns
        ------------
        - None
        '''
        fig, axes = plt.subplots(2, 3, figsize=(30, 15), sharex=True)
        axes = axes.flatten()

        if title == "TruncatedSVD":
            model = model.components_ # components_ es la matriz VT
        else:
            model = model
        for topic_idx, topic in enumerate(model):

                #Devuelve los índices de los elementos que ordenan el arreglo de menor a mayor
                top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
                #devuelve la palabra basada en el índice
                top_features = [feature_names[i] for i in top_features_ind]

                weights = topic[top_features_ind]

                ax = axes[topic_idx]
                ax.barh(top_features, weights, height=0.7)

                ax.invert_yaxis()
                ax.tick_params(axis="both", which="major", labelsize=17)
        plt.subplots_adjust(top=0.98, bottom=0.04, wspace=0.23, hspace=0.15)

def main():
    # ---------------------------------------------------------
    # Reading all the documents with the excel extension (xlsx)
    # in the Survey-Results folder
    # ---------------------------------------------------------
    file_names = os.listdir('../Survey-Results')

    data_from_files = []
    names = []

    print('Leyendo los archivos de los webinars...')
    for file in file_names:
        if file.endswith('.xlsx'):
            names.append(file)
            data = pd.read_excel('../Survey-Results/' + file,
                                sheet_name='NLP')
            data_from_files.append(data)
    print('Archivos leídos correctamente')
    
    total_opinions = np.sum([len(data_from_files[i]) for i in range(len(data_from_files))])
    total_webinars = len(names)
    
    # Loading the Spanish model
    nlp = spacy.load("es_core_news_sm")

    # ----------------------------------
    # Writing the report
    # ----------------------------------
    # Number of opinions by type

    print('Generando el PDF...')
    # Create a PDF file
    pdf_path = f'../Results/Areas de Oportunidad.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    # Create a list to hold the story (elements to be added to the PDF)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    title_style = styles['Title']
    heading_style = styles['Heading2']
    # Crear un estilo personalizado con alineación justificada
    custom_style = ParagraphStyle(name="CustomStyle", parent=styles["Normal"], alignment=TA_JUSTIFY)

    # Add the current date and author
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    author = 'Axel Daniel Malváez Flores' 
    date_paragraph = Paragraph(f'<b>Fecha del Informe:</b> {current_date}', normal_style)
    author_paragraph = Paragraph(f'<b>Autor:</b> {author}', normal_style)

    story.append(date_paragraph)
    story.append(author_paragraph)
    story.append(Spacer(1, 12))

    # Add a title to the PDF
    title = Paragraph(f'Informe de las Áreas de Oportunidad en los futuros Webinars', title_style)
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Introduction Seccion
    introduction = f'''
    Este informe es un análisis de las áreas de oportunidad para los futuros webinars que se presenten en la
    Fundación Academia Aesculap México. El objetivo es proporcionar una visión general de las opiniones de los
    participantes y de las posibles áreas dentro del sector de la salud que podrían ser de interés para los participantes.
    Link plataforma :<u>http://academiaaesculap.eadbox.com/</u>
    '''
    introduction_paragraph = Paragraph(introduction, custom_style)
    story.append(introduction_paragraph)
    story.append(Spacer(1, 12))
    
    # Methodology Seccion
    methodology_title = 'Metodología'
    methodology = f'''
    Para llevar a cabo este análisis, se recopilaron los resultados de las encuestas llevadas a cabo en los más recientes webinars.
    Estos resultados fueron analizados utilizando técnicas de Procesamiento de Lenguaje Natural (PLN). En primer lugar, se
    limpiaron para eliminar caracteres especiales y palabras vacías, posteriormente se corrigieron errores ortográficos y se
    convirtieron todas las palabras a minúsculas. Finalmente, se cuentan las palabras más comunes en los resultados de la columna
    <i>¿Qué es lo que más te gusta?</i> para ver los temas que más interesan a los participantes. Por otro lado para la parte de áreas 
    de oportunidad se utilizó la técnica de LSA (Latent Semantic Analysis) para encontrar los temas más comunes en los resultados de la
    columna <i>¿Qué temas te gustaría que se abordaran en futuros webinars?</i>. 
    '''
    methodology_paragraph = Paragraph(methodology_title, heading_style)
    methodology_content = Paragraph(methodology, custom_style)
    story.append(methodology_paragraph)
    story.append(methodology_content)
    story.append(Spacer(1, 12))
    
    # Results Seccion for liked topics
    results_title = 'Resultados ¿Qué es lo que más te gusta?'
    
    results = f'''
    Los siguientes resultados son las palabras más comunes en las respuestas a la pregunta<i>¿Qué es lo que más te gusta?</i>. Cada
    gráfica representa los resultados de un webinar en particular. En total, se han evaluado {total_opinions}respuestas de
    {total_webinars} webinars. Finalmente se presenta una gráfica que engloba los resultados de todos los webinars.
    '''
    results_paragraph = Paragraph(results_title, heading_style)
    results_content = Paragraph(results, custom_style)
    story.append(results_paragraph)
    story.append(results_content)
    story.append(Spacer(1, 12))

    original_width = 800
    original_height = 500
    scale_factor = 0.5
    
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    print('Generando las gráficas de los resultados más gustados...')
    # ----------------------------------
    # NLP for liked topics
    # ----------------------------------

    words_global = []

    for i in range(len(data_from_files)):
        # Selecting Data
        liked = data_from_files[i].iloc[:,0]
        
        data_non_stop = []
        words = []
        
        data_non_stop = [remove_stop_words(op.lower(), nlp(op)) for op in liked]
        words_documents = [x.split() for x in data_non_stop]
        words = [item for sublist in words_documents for item in sublist]
                    
        words_global.extend(words)
            
        words_counted = Counter(words)
        most_common = words_counted.most_common(10)

        plt.figure(figsize=(8, 5))
        sns.barplot(y=[x[0] for x in most_common], x=[x[1] for x in most_common])
        for p in plt.gca().patches:
            plt.gca().annotate(int(p.get_width()), (p.get_x() + p.get_width(), p.get_y() + 0.8), xytext=(5, 10), textcoords='offset points')
        
        plt.title(f"Most common words in {names[i]}")
        plt.savefig(f"../Results/Most_common_words_in_{names[i]}.png")
        
        statistics = f'''
        En el webinar {names[i]} las palabras más comunes son:
        '''
        statistics_paragraph = Paragraph(statistics, custom_style)
        story.append(statistics_paragraph)

        image = Image(f'../Results/Most_common_words_in_{names[i]}.png', width=new_width, height=new_height)    
        story.append(image)
        story.append(Spacer(1, 12))
    
    print('Generando las gráficas de los resultados más gustados global...')

    # Global results for liked topics
    words_counted = Counter(words_global)
    most_common = words_counted.most_common(10)

    plt.figure(figsize=(8, 5))
    sns.barplot(y=[x[0] for x in most_common], x=[x[1] for x in most_common])
    for p in plt.gca().patches:
        plt.gca().annotate(int(p.get_width()), (p.get_x() + p.get_width(), p.get_y() + 0.8), xytext=(5, 10), textcoords='offset points')
    
    plt.title(f"Most common words in all webinars")
    plt.savefig(f"../Results/Most_common_words_in_all_webinars.png")
    
    statistics = f'''
    En todos los webinars las palabras más comunes son:
    {most_common}
    '''
    statistics_paragraph = Paragraph(statistics, custom_style)
    story.append(statistics_paragraph)
    
    image = Image('../Results/Most_common_words_in_all_webinars.png', width=new_width, height=new_height)
    story.append(image)
    story.append(Spacer(1, 12))
    
    print('Gráficas generadas correctamente')
    
    # Results Seccion for recommended topics
    results_title_2 = 'Resultados ¿Qué temas te gustaría que se abordaran en futuros webinars?'
    
    results_2 = f'''
    Los siguientes resultados son los temas más comunes en las respuestas a la pregunta<i>¿Qué temas te gustaría que se abordaran
    en futuros webinars?</i>. Cada gráfica nos representa los diferentes tópicos que nuestras personas encuestadas nos han mencionado.
    Es necesario enfatizar que los resultados se hicieron suponiendo que existen 6 diferentes tópicos, por lo que es posible que
    existan más tópicos de los que se muestran en las gráficas. No obstante, podríamos considerar que dichos tópicos son los más comunes. 
    Estas gráficas no cuentan la frecuencia de las palabras, sino que le asignan un peso a cada palabra dependiendo de la importancia
    que tiene en cada una de las opiniones.
    '''
    results_paragraph_2 = Paragraph(results_title_2, heading_style)
    results_content_2 = Paragraph(results_2, custom_style)
    story.append(results_paragraph_2)
    story.append(results_content_2)

    print('Generando las gráficas de los resultados más recomendados...')

    # ----------------------------------
    # NLP for recommended topics
    # ----------------------------------
    interests = [ i for j in range(len(data_from_files)) for i in data_from_files[j].iloc[:,1].dropna() ]
    interests_non_stop = [unidecode(remove_stop_words(op.lower(), nlp(op))) for op in interests]

    # Spell checker
    spell = SpellChecker(language='es')

    # Remove special characters
    interests_cleaned = [ str(i).replace("[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ]", " ").strip() for i in interests_non_stop]

    # Correct spelling
    interests_cleaned_and_corrected = []
    for inter in interests_cleaned:
        words = inter.split()
        corrected_words = []
        for word in words:
            corrected_word = spell.correction(word)
            if word == 'covid' or word == 'onco':
                corrected_words.append(word)
                continue
            if word == 'obtetricas':
                corrected_words.append('obstétricas')
                continue
            if corrected_word != None:
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        corrected_text = ' '.join(corrected_words)
        interests_cleaned_and_corrected.append(corrected_text)

    # Dataframe with the interests and the interests cleaned and corrected
    pre_data = pd.DataFrame({'interests':interests, 'interests_cleaned':interests_cleaned, 'interests_cleaned_and_corrected':interests_cleaned_and_corrected})

    # Building the TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features= 1000, # maximum number of words
                                max_df = 0.95,
                                min_df=2,
                                smooth_idf=True)

    # Transforming the data
    X = vectorizer.fit_transform(pre_data['interests_cleaned_and_corrected'])

    # Getting the words
    terminos = vectorizer.get_feature_names_out()

    # Applying TruncatedSVD
    svd_model = TruncatedSVD(n_components=6, #Dimensionalidad deseada de los datos de salida.
                            algorithm='randomized',
                            n_iter=100, random_state=122)
    svd_model.fit_transform(X)

    # Plotting the results with 7 words and 6 topics
    plot_top_words(svd_model, terminos, 7, "TruncatedSVD")

    # Saving the plot
    plt.savefig(f"../Results/TruncatedSVD.png", dpi=300, bbox_inches='tight')
    
    print('Gráficas generadas correctamente')

    original_width = 7402
    original_height = 4384
    scale_factor = 0.8
    image = Image('../Results/TruncatedSVD.png', width=new_width, height=new_height)

    story.append(image)
    story.append(Spacer(1, 12))    
    
    # Recomendaciones
    recommendations_title = 'Recomendaciones'
    recommendations = '''
    Finalmente, se recomienda poner atención a lo que más les gustó a los participantes de los webinars, pues podríamos hacer esta
    pregunta en las encuestas de los próximos webinars de manera categorizada para obtener información más precisa basándonos en las
    respuestas de los participantes más comunes. Por otro lado, se recomienda poner atención a los temas que les gustaría que se
    abordaran en futuros webinars, esto para poder mejorar la calidad de los webinars y que sean de mayor interés para los participantes.
    '''
    recommendations_paragraph = Paragraph(recommendations_title, heading_style)
    recommendations_content = Paragraph(recommendations, custom_style)
    story.append(recommendations_paragraph)
    story.append(recommendations_content)
    
    # Build the PDF
    doc.build(story)
    print(f'PDF generado como : {pdf_path}')
    
if __name__ == "__main__":
    main()