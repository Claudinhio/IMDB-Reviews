#!/usr/bin/env python
# coding: utf-8

# Film Junky Union, una nueva comunidad vanguardista para los aficionados de las películas clásicas, está desarrollando un sistema para filtrar y categorizar reseñas de películas. Tu objetivo es entrenar un modelo para detectar las críticas negativas de forma automática. Para lograrlo, utilizarás un conjunto de datos de reseñas de películas de IMDB con leyendas de polaridad para construir un modelo para clasificar las reseñas positivas y negativas. Este deberá alcanzar un valor F1 de al menos 0.85.
# 
# ## Inicialización

# In[1]:


import math
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sklearn.metrics as metrics
import torch
import transformers
import nltk
import spacy
import tensorflow as tf

from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from tqdm import tqdm
from tqdm.auto import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
# la siguiente línea proporciona gráficos de mejor calidad en pantallas HiDPI
# %config InlineBackend.figure_format = 'retina'

plt.style.use('seaborn')


# In[3]:


# esto es para usar progress_apply, puedes leer más en https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()


# ## Cargar datos

# In[4]:


df_reviews = pd.read_csv('/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})


# In[5]:


display(df_reviews.head())
display(df_reviews.info())


# In[6]:


# Paso 2: Preprocesar los datos
# Eliminamos valores nulos y revisamos la estructura de los datos
df_reviews.dropna(inplace=True)
print(df_reviews.info())


# In[7]:


# Función de limpieza de texto
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Elimina HTML
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Elimina caracteres especiales
    text = text.lower()  # Convierte a minúsculas
    return text

df_reviews['review'] = df_reviews['review'].apply(clean_text)

display(df_reviews.head())
display(df_reviews.info())


# ## EDA
# 
# El código genera dos gráficos de barras que representan la evolución de las películas y las reseñas a lo largo de los años:
# 
# - Número de películas a lo largo de los años: Este gráfico muestra la cantidad de películas únicas (tconst) por año (start_year). El código primero elimina los duplicados y luego cuenta cuántas películas hay por año. Luego, rellena los años sin películas con cero. Finalmente, se traza un gráfico de barras para visualizar la distribución.
# - Número de reseñas a lo largo de los años: Este gráfico es un poco más complejo ya que muestra dos tipos de datos. Primero, se traza un gráfico de barras apiladas que muestra el número de reseñas negativas y positivas (pos) por año. Luego, se calcula el número total de reseñas por año y se divide por el número de películas para obtener el promedio de reseñas por película. Este promedio se traza como una línea naranja en el gráfico. La línea representa un promedio móvil de 5 años, lo que ayuda a suavizar las fluctuaciones anuales y a mostrar la tendencia general.
# 
# Estos gráficos podrían proporcionar información valiosa sobre cómo ha evolucionado la industria del cine a lo largo del tiempo en términos de producción de películas y participación del público. Por ejemplo, si el número de películas y reseñas está aumentando con el tiempo, esto podría indicar un crecimiento en la industria y un mayor compromiso del público. Además, la relación entre el número de reseñas y el número de películas podría dar una idea de cómo ha cambiado la proporción de reseñas por película a lo largo del tiempo.
# 
# Veamos el número de películas y reseñas a lo largo de los años.

# In[8]:


fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Número de películas a lo largo de los años')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Número de reseñas a lo largo de los años')

fig.tight_layout()


# - Gráfico de barras de #Reseñas por película: Este gráfico muestra la cantidad de reseñas por película. El código primero agrupa el dataframe df_reviews por película (tconst) y cuenta el número de reseñas (review). Luego, cuenta cuántas películas tienen un cierto número de reseñas y ordena el resultado. Finalmente, se traza un gráfico de barras para visualizar la distribución.
# - Gráfico KDE de #Reseñas por película: Este gráfico muestra la distribución de densidad del número de reseñas por película. El código es similar al anterior, pero en lugar de contar cuántas películas tienen un cierto número de reseñas, simplemente agrupa el dataframe por película y cuenta las reseñas. Luego, se utiliza la función kdeplot de seaborn para trazar la distribución de densidad del número de reseñas.
# 
# Estos gráficos podrían proporcionar información valiosa sobre cómo se distribuyen las reseñas entre las películas. Por ejemplo, la mayoría de las películas tienen pocas reseñas, esto podría indicar que solo unas pocas películas reciben la mayoría de las reseñas. Por otro lado, el gráfico de densidad muestra una distribución más uniforme, esto podría indicar que las reseñas están más equitativamente distribuidas entre las películas.

# In[9]:


fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Gráfico de barras de #Reseñas por película')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('Gráfico KDE de #Reseñas por película')

fig.tight_layout()


# - Distribución de Reseñas Positivas y Negativas: Este gráfico muestra la cantidad de reseñas positivas y negativas en el conjunto de datos df_reviews. El código utiliza la función countplot de seaborn para crear un gráfico de barras donde el eje x representa las clases de reseñas (positivas y negativas) y el eje y representa el conteo de cada clase.
# - Porcentajes de cada clase: Este fragmento de código calcula el porcentaje de reseñas positivas y negativas en el conjunto de datos. Primero, cuenta el número de reseñas positivas y negativas. Luego, normaliza estos conteos para obtener los porcentajes y los multiplica por 100 para obtener los porcentajes en una escala de 0 a 100. Finalmente, imprime estos porcentajes.
# 
# En cuanto a las conclusiones generales, los resultados muestran que las reseñas están casi igualmente distribuidas entre positivas y negativas, con un 50.1067% de reseñas negativas y un 49.8933% de reseñas positivas. Esto sugiere que el conjunto de datos está equilibrado en términos de la variable objetivo (pos), lo cual es una buena noticia si planeas utilizar estos datos para entrenar un modelo de clasificación, ya que los modelos tienden a funcionar mejor cuando las clases están equilibradas.

# In[10]:


# Análisis exploratorio de datos
sns.countplot(x='pos', data=df_reviews)
plt.title('Distribución de Reseñas Positivas y Negativas')
plt.show()

# Mostrar porcentajes de cada clase
class_counts = df_reviews['pos'].value_counts(normalize=True) * 100
print(class_counts)


# In[11]:


df_reviews['pos'].value_counts()


# - El conjunto de entrenamiento: distribución de puntuaciones: Este gráfico muestra la distribución de las puntuaciones de las reseñas en el conjunto de entrenamiento. El código primero filtra las reseñas del conjunto de entrenamiento, luego cuenta cuántas reseñas hay para cada puntuación y ordena el resultado. Luego, rellena las puntuaciones faltantes con cero. Finalmente, se traza un gráfico de barras para visualizar la distribución.
# - El conjunto de prueba: distribución de puntuaciones: Este gráfico es similar al anterior, pero para el conjunto de prueba.
# 
# En cuanto a las conclusiones generales, estos gráficos podrían proporcionar información valiosa sobre cómo se distribuyen las puntuaciones de las reseñas en los conjuntos de entrenamiento y prueba. Esto es importante para entender si los conjuntos de entrenamiento y prueba tienen distribuciones similares, lo cual es deseable para obtener un modelo de aprendizaje automático que se desempeñe bien.

# In[12]:


fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de entrenamiento: distribución de puntuaciones')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de prueba: distribución de puntuaciones')

fig.tight_layout()


# El código genera cuatro gráficos que representan la distribución de las reseñas positivas y negativas en los conjuntos de entrenamiento y prueba, tanto por año como por película. Aquí te dejo algunos comentarios sobre el código:
# 
# - El conjunto de entrenamiento: número de reseñas de diferentes polaridades por año: Este gráfico muestra la cantidad de reseñas positivas y negativas en el conjunto de entrenamiento por año. El código primero filtra las reseñas del conjunto de entrenamiento, luego agrupa las reseñas por año y polaridad, y cuenta cuántas reseñas hay para cada combinación. Luego, rellena los años sin reseñas con cero. Finalmente, se traza un gráfico de barras apiladas para visualizar la distribución.
# - El conjunto de entrenamiento: distribución de diferentes polaridades por película: Este gráfico muestra la distribución de las reseñas positivas y negativas en el conjunto de entrenamiento por película. El código es similar al anterior, pero en lugar de agrupar por año, agrupa por película. Luego, se utiliza la función kdeplot de seaborn para trazar la distribución de densidad de las reseñas.
# - El conjunto de prueba: número de reseñas de diferentes polaridades por año: Este gráfico es similar al primero, pero para el conjunto de prueba.
# - El conjunto de prueba: distribución de diferentes polaridades por película: Este gráfico es similar al segundo, pero para el conjunto de prueba.

# In[13]:


fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de entrenamiento: número de reseñas de diferentes polaridades por año')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de entrenamiento: distribución de diferentes polaridades por película')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de prueba: número de reseñas de diferentes polaridades por año')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de prueba: distribución de diferentes polaridades por película')

fig.tight_layout()


# En general, estas gráficas proporcionan una visión útil de cómo ha evolucionado el conjunto de datos de entrenamiento con el tiempo y cómo ha cambiado la distribución de las polaridades. Esto puede ser útil para entender cómo está funcionando el modelo y dónde puede haber oportunidades para mejorarlo.

# ## Procedimiento de evaluación
# 
# Composición de una rutina de evaluación que se pueda usar para todos los modelos en este proyecto.
# 
# La función evaluate_model, toma como entrada un modelo y conjuntos de características y objetivos tanto para entrenamiento como para prueba:
# 
# - Predicciones: Para cada conjunto de datos (entrenamiento y prueba), la función genera predicciones de clase (pred_target) y probabilidades de la clase positiva (pred_proba) utilizando el modelo proporcionado.
# - Valor F1: La función calcula el valor F1 para diferentes umbrales de clasificación y los traza en un gráfico. Esto puede ser útil para seleccionar un umbral de clasificación que maximice el valor F1.
# - Curva ROC: La función traza la curva ROC, que es un gráfico de la tasa de verdaderos positivos frente a la tasa de falsos positivos para diferentes umbrales de clasificación. También calcula el área bajo la curva ROC (ROC AUC), que es una medida de la capacidad del modelo para distinguir entre las clases.
# - Curva PRC: La función traza la curva de precisión-recall, que es un gráfico de la precisión frente al recall para diferentes umbrales de clasificación. También calcula la puntuación promedio de precisión (APS), que es una medida de la capacidad del modelo para predecir la clase positiva.
# - Exactitud y F1: La función calcula la exactitud y el valor F1 del modelo utilizando las predicciones de clase.
# - Estadísticas de evaluación: Finalmente, la función imprime un dataframe que resume las estadísticas de evaluación para los conjuntos de entrenamiento y prueba.
# 
# Esta función es una herramienta útil para evaluar y comparar diferentes modelos de aprendizaje automático en términos de varias métricas importantes. 

# In[14]:


def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'Curva ROC')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Exactitud'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Exactitud', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return


# ## Normalización
# 
# Suponemos que todos los modelos a continuación aceptan textos en minúsculas y sin dígitos, signos de puntuación, etc.

# In[15]:


df_reviews['review_norm'] = df_reviews['review']


# ## División entrenamiento / prueba
# 
# Por fortuna, todo el conjunto de datos ya está dividido en partes de entrenamiento/prueba; 'ds_part' es el indicador correspondiente.

# In[16]:


df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)


# El conjunto de entrenamiento (df_reviews_train) tiene 23796 filas y 18 columnas, y se utilizará para entrenar los modelos. Por otro lado, el conjunto de prueba (df_reviews_test) tiene 23533 filas y 18 columnas, y se utilizará para evaluar el rendimiento de los modelos en datos no vistos.
# 
# Las variables train_target y test_target contienen las etiquetas de las reseñas (positivas o negativas) para los conjuntos de entrenamiento y prueba, respectivamente.
# 
# Este es un buen paso en tu flujo de trabajo de ciencia de datos, ya que asegura que los modelos no se evalúen con los mismos datos que se utilizaron para entrenarlos, lo cual es importante para obtener una evaluación justa y precisa del rendimiento del modelo.

# ## Trabajar con modelos
# 
# 
# 
# 

# ### Modelo Constante Dummy

# In[17]:


# Crear y entrenar el DummyClassifier
dummy_classifier = DummyClassifier(strategy="constant", constant=1)
dummy_classifier.fit(df_reviews_train[['pos']], train_target)

# Evaluar el DummyClassifier
print("Evaluación del DummyClassifier:")
evaluate_model(dummy_classifier, df_reviews_train[['pos']], train_target, df_reviews_test[['pos']], test_target)


# Los resultados indican el rendimiento del modelo DummyClassifier en el conjunto de entrenamiento y prueba.
# 
# - Exactitud: La exactitud del modelo en el conjunto de entrenamiento y en el conjunto de prueba es 0.5. Esto significa que el modelo clasifica correctamente el 50% de las reseñas en ambos conjuntos. Dado que este es un modelo Dummy que siempre predice la clase más frecuente, esta exactitud del 50% indica que las clases están balanceadas en tus datos.
# 
# - F1-score: El valor F1 es 0.67 tanto en el conjunto de entrenamiento como en el de prueba. Esto sugiere que el modelo tiene un equilibrio razonable entre precisión y recall.
# 
# - APS (Average Precision Score): Este puntaje es 0.5 tanto en el conjunto de entrenamiento como en el de prueba, lo que indica que el modelo tiene un rendimiento aleatorio en la clasificación de reseñas positivas y negativas.
# 
# - ROC AUC: El área bajo la curva ROC es 0.5 tanto en el conjunto de entrenamiento como en el de prueba. Un valor de 0.5 indica que el modelo no tiene capacidad de discriminación para distinguir entre reseñas positivas y negativas, lo cual es esperado para un modelo Dummy.
# 
# En resumen, el DummyClassifier no es efectivo para clasificar reseñas de películas en positivas y negativas, En resumen, aunque hemos ajustado el modelo, sigue siendo un DummyClassifier sin capacidad real de aprendizaje.

# El código siguiente realiza varias tareas de preprocesamiento de texto y vectorización para un conjunto de datos de reseñas de películas.
# 
# - Descarga de recursos de NLTK: Este fragmento de código descarga los recursos necesarios de NLTK, que es una biblioteca de Python para el procesamiento del lenguaje natural.
# - Preprocesamiento de texto con NLTK: Esta función toma un texto como entrada, lo tokeniza, elimina las palabras vacías y las palabras de una sola letra, y luego une los tokens en un solo texto.
# - Preprocesamiento de texto con spaCy: Esta función es similar a la anterior, pero utiliza la biblioteca spaCy para el preprocesamiento. En lugar de simplemente tokenizar el texto y eliminar las palabras vacías, esta función también lematiza los tokens, lo que significa que reduce las palabras a su forma base o raíz.
# - Aplicación del preprocesamiento a las reseñas: Este fragmento de código aplica las funciones de preprocesamiento a las reseñas en los conjuntos de entrenamiento y prueba.
# - TF-IDF: Finalmente, este fragmento de código inicializa dos vectorizadores TF-IDF, que pueden transformar el texto en vectores numéricos que pueden ser utilizados por los modelos de aprendizaje automático. El primer vectorizador limita el número de características a 1000, mientras que el segundo lo limita a 100 para evitar la dimensionalidad excesiva.
# 
# Es importante recordar que el preprocesamiento de texto puede variar dependiendo del problema específico y del conjunto de datos. Por ejemplo, en algunos casos, podría ser útil mantener las palabras de una sola letra, especialmente si esas palabras tienen un significado importante en el contexto del problema. Además, la lematización puede no siempre ser deseable, ya que puede cambiar el significado de algunas palabras. Por último, la elección del número de características en la vectorización TF-IDF puede tener un impacto significativo en el rendimiento del modelo y debe ser ajustada cuidadosamente.

# In[17]:


# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Preprocesamiento de texto con NLTK
stop_words = set(stopwords.words('english'))

def preprocess_text_nltk(text):
    # Tokenización
    tokens = nltk.word_tokenize(text)
    # Eliminación de stopwords y palabras de una sola letra
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and len(word) > 1]
    # Reunir tokens nuevamente en texto
    return ' '.join(filtered_tokens)

# Preprocesamiento de texto con spaCy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def preprocess_text_spacy(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return ' '.join(tokens)

# Aplicar preprocesamiento a las reseñas
df_reviews_train['review_processed_nltk'] = df_reviews_train['review_norm'].progress_apply(preprocess_text_nltk)
df_reviews_test['review_processed_nltk'] = df_reviews_test['review_norm'].progress_apply(preprocess_text_nltk)

df_reviews_train['review_processed_spacy'] = df_reviews_train['review_norm'].progress_apply(preprocess_text_spacy)
df_reviews_test['review_processed_spacy'] = df_reviews_test['review_norm'].progress_apply(preprocess_text_spacy)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_vectorizer2 = TfidfVectorizer(max_features=100) # Limita el número de características para evitar la dimensionalidad excesiva


# ### Modelo NLTK, TF-IDF y LR

# In[27]:


# Para NLTK
train_features_tfidf_nltk = tfidf_vectorizer.fit_transform(df_reviews_train['review_processed_nltk'])
test_features_tfidf_nltk = tfidf_vectorizer.transform(df_reviews_test['review_processed_nltk'])

# Modelo de Regresión Logística
logistic_regression_model_nltk = LogisticRegression()
logistic_regression_model_nltk.fit(train_features_tfidf_nltk, train_target)

# Evaluación del modelo
print("Evaluación del modelo de Regresión Logística con preprocesamiento NLTK:")
evaluate_model(logistic_regression_model_nltk, train_features_tfidf_nltk, train_target, test_features_tfidf_nltk, test_target)


# Los resultados  indican el rendimiento del modelo de Regresión Logística en el conjunto de entrenamiento y prueba. Aquí está la interpretación:
# 
# - Exactitud: La exactitud del modelo en el conjunto de entrenamiento es 0.87 y en el conjunto de prueba es 0.86. Esto significa que el modelo clasifica correctamente el 87% de las reseñas en el conjunto de entrenamiento y el 86% en el conjunto de prueba.
# 
# - F1-score: El valor F1 mide la precisión del modelo, teniendo en cuenta tanto la precisión como el recall. Un puntaje de 87% en el conjunto de entrenamiento y 86% en el conjunto de prueba sugiere que el modelo tiene un buen equilibrio entre precisión y recall.
# 
# - APS (Average Precision Score): Este puntaje mide la calidad del modelo basado en la curva de precisión-recall. Un valor de 94% en el conjunto de entrenamiento y en el conjunto de prueba indica que el modelo tiene un buen rendimiento en la clasificación de reseñas positivas y negativas.
# 
# - ROC AUC: El área bajo la curva ROC es otra métrica de rendimiento común para modelos de clasificación binaria. Un valor de 95% en el conjunto de entrenamiento y en el conjunto de prueba 94% sugiere que el modelo tiene un buen poder de discriminación para distinguir entre reseñas positivas y negativas.
# 
# En general, estos resultados sugieren que el modelo de Regresión Logística funciona bien para clasificar reseñas de películas en positivas y negativas. Sin embargo, hay que tener en cuenta que al reducir el número de características a 1000 en el vectorizador TF-IDF, se puede estar perdiendo información relevante para la clasificación. Es posible que experimentar con diferentes valores para max_features pueda mejorar aún más el rendimiento del modelo. 

# ### Modelo spaCy, TF-IDF y LR

# In[28]:


# Para spaCy
train_features_tfidf_spacy = tfidf_vectorizer.fit_transform(df_reviews_train['review_processed_spacy'])
test_features_tfidf_spacy = tfidf_vectorizer.transform(df_reviews_test['review_processed_spacy'])

# Modelo de Regresión Logística

logistic_regression_model_spacy = LogisticRegression()
logistic_regression_model_spacy.fit(train_features_tfidf_spacy, train_target)

print("Evaluación del modelo de Regresión Logística con preprocesamiento spaCy:")
evaluate_model(logistic_regression_model_spacy, train_features_tfidf_spacy, train_target, test_features_tfidf_spacy, test_target)


# Los resultados muestran que el modelo de Regresión Logística con preprocesamiento spaCy tiene un rendimiento bastante bueno, con una exactitud y un valor F1 de alrededor del 87% en el conjunto de prueba. Sin embargo, siempre es una buena idea comparar estos resultados con los de otros modelos o enfoques de preprocesamiento para asegurarse de que este modelo es el más adecuado para el problema específico que estás tratando de resolver.

# ### Modelo spaCy, TF-IDF y LGBMClassifier

# In[29]:


# Modelo LGBMClassifier
lgbm_classifier_spacy = LGBMClassifier()
lgbm_classifier_spacy.fit(train_features_tfidf_spacy, train_target)

print("Evaluación del modelo de LGBMClassifier con preprocesamiento spaCy:")
# Evaluación del modelo
evaluate_model(lgbm_classifier_spacy, train_features_tfidf_spacy, train_target, test_features_tfidf_spacy, test_target)


# - Exactitud: La exactitud del modelo en el conjunto de entrenamiento es 0.92 y en el conjunto de prueba es 0.87. Esto significa que el modelo clasifica correctamente el 92% de las reseñas en el conjunto de entrenamiento y el 87% en el conjunto de prueba.
# 
# - F1-score: El valor F1 mide la precisión del modelo, teniendo en cuenta tanto la precisión como el recall. Un puntaje de 92% en el conjunto de entrenamiento y 87% en el conjunto de prueba sugiere que el modelo tiene un buen equilibrio entre precisión y recall.
# 
# - APS (Average Precision Score): Este puntaje mide la calidad del modelo basado en la curva de precisión-recall. Un valor de 98% en el conjunto de entrenamiento y 94% en el conjunto de prueba indica que el modelo tiene un buen rendimiento en la clasificación de reseñas positivas y negativas.
# 
# - ROC AUC: El área bajo la curva ROC es otra métrica de rendimiento común para modelos de clasificación binaria. Un valor de 98% en el conjunto de entrenamiento y 94% en el conjunto de prueba sugiere que el modelo tiene un buen poder de discriminación para distinguir entre reseñas positivas y negativas.
# 
# En general, estos resultados sugieren que el modelo LGBMClassifier funciona bien para clasificar reseñas de películas en positivas y negativas. Sin embargo, hay una ligera disminución en el rendimiento del modelo en el conjunto de prueba en comparación con el conjunto de entrenamiento, lo que podría indicar un cierto grado de sobreajuste. Podría ser útil experimentar con diferentes parámetros del modelo para mejorar su capacidad de generalización.

# ### Modelo NLTK, TF-IDF y LGBMClassifier

# In[30]:


# Modelo LGBMClassifier
lgbm_classifier_nltk = LGBMClassifier()
lgbm_classifier_nltk.fit(train_features_tfidf_nltk, train_target)

print("Evaluación del modelo de LGBMClassifier con preprocesamiento NLTK:")
# Evaluación del modelo
evaluate_model(lgbm_classifier_nltk, train_features_tfidf_nltk, train_target, test_features_tfidf_nltk, test_target)


# Los resultados muestran que el modelo de LGBMClassifier con preprocesamiento NLTK tiene un rendimiento bastante bueno, con un valor F1 de alrededor del 85% en el conjunto de prueba. 

# 

# In[31]:


# SVM (Support Vector Machines):
#from sklearn.svm import SVC

# Modelo SVM con preprocesamiento NLTK
#svm_model_nltk = SVC(probability=True)
#svm_model_nltk.fit(train_features_tfidf_nltk, train_target)

# Evaluación del modelo
#print("Evaluación del modelo SVM con preprocesamiento NLTK:")
#evaluate_model(svm_model_nltk, train_features_tfidf_nltk, train_target, test_features_tfidf_nltk, test_target)

# Modelo SVM con preprocesamiento spaCy
#svm_model_spacy = SVC(probability=True)
#svm_model_spacy.fit(train_features_tfidf_spacy, train_target)

# Evaluación del modelo
#print("Evaluación del modelo SVM con preprocesamiento spaCy:")
#evaluate_model(svm_model_spacy, train_features_tfidf_spacy, train_target, test_features_tfidf_spacy, test_target)


# ### Modelo Árbol de Decisión

# In[32]:


# Modelo de Árbol de Decisión
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(train_features_tfidf_nltk, train_target)
# Evaluación del modelo
print("Evaluación del modelo Árbol de Decisión con preprocesamiento NLTK:")
# Evaluación del modelo
evaluate_model(decision_tree_model, train_features_tfidf_nltk, train_target, test_features_tfidf_nltk, test_target)

# Modelo  Árbol de Decisión con preprocesamiento spaCy
decision_tree_model_spacy = DecisionTreeClassifier()
decision_tree_model_spacy.fit(train_features_tfidf_spacy, train_target)

# Evaluación del modelo
print("Evaluación del modelo Árbol de Decisión con preprocesamiento spaCy:")
evaluate_model(decision_tree_model_spacy, train_features_tfidf_spacy, train_target, test_features_tfidf_spacy, test_target)


# - Árbol de Decisión con preprocesamiento NLTK: Este modelo tiene un rendimiento perfecto en el conjunto de entrenamiento, con una exactitud, un valor F1, una puntuación promedio de precisión (APS) y un área bajo la curva ROC (ROC AUC) de 1.0. Sin embargo, su rendimiento disminuye en el conjunto de prueba, con una exactitud, un valor F1 y un ROC AUC de 0.70, y una APS de 0.64.
# - Árbol de Decisión con preprocesamiento spaCy: Este modelo tiene un rendimiento idéntico al del modelo con preprocesamiento NLTK.
# 
# Estos resultados sugieren que ambos modelos están sobreajustando al conjunto de entrenamiento, ya que su rendimiento es perfecto en el conjunto de entrenamiento pero disminuye significativamente en el conjunto de prueba. Esto es común en los modelos de Árbol de Decisión, que tienden a crear árboles muy complejos que se ajustan perfectamente a los datos de entrenamiento pero no generalizan bien a nuevos datos. 

# ### Modelo Bosque Aleatorio

# In[33]:


# Modelo Random Forest con preprocesamiento NLTK
random_forest_model_nltk = RandomForestClassifier()
random_forest_model_nltk.fit(train_features_tfidf_nltk, train_target)

# Evaluación del modelo
print("Evaluación del modelo Random Forest con preprocesamiento NLTK:")
evaluate_model(random_forest_model_nltk, train_features_tfidf_nltk, train_target, test_features_tfidf_nltk, test_target)

# Modelo Random Forest con preprocesamiento spaCy
random_forest_model_spacy = RandomForestClassifier()
random_forest_model_spacy.fit(train_features_tfidf_spacy, train_target)

# Evaluación del modelo
print("Evaluación del modelo Random Forest con preprocesamiento spaCy:")
evaluate_model(random_forest_model_spacy, train_features_tfidf_spacy, train_target, test_features_tfidf_spacy, test_target)


# - Bosques Aleatorios con preprocesamiento NLTK: Este modelo tiene un rendimiento perfecto en el conjunto de entrenamiento, con una exactitud, un valor F1, una puntuación promedio de precisión (APS) y un área bajo la curva ROC (ROC AUC) de 1.0. Sin embargo, su rendimiento disminuye en el conjunto de prueba, con una exactitud y un valor F1 de 0.83, y una APS y un ROC AUC de 0.90 y 0.91 respectivamente.
# - Bosques Aleatorios con preprocesamiento spaCy: Este modelo tiene un rendimiento idéntico al del modelo con preprocesamiento NLTK en el conjunto de entrenamiento. En el conjunto de prueba, su rendimiento es ligeramente inferior, con una exactitud y un valor F1 de 0.82, y una APS y un ROC AUC de 0.90 y 0.91 respectivamente.

# In[ ]:


# Modelo XGBoost
# from xgboost import XGBClassifier

# Modelo XGBoost
# xgb_model_nltk = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# xgb_model_nltk.fit(train_features_tfidf_nltk, train_target)

# xgb_model_spacy = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# xgb_model_spacy.fit(train_features_tfidf_spacy, train_target)

# Evaluación del modelo
# print("Evaluación del modelo XGBoost con preprocesamiento NLTK:")
# evaluate_model(xgb_model_nltk, train_features_tfidf_nltk, train_target, test_features_tfidf_nltk, test_target)

# print("Evaluación del modelo XGBoost con preprocesamiento spaCy:")
# evaluate_model(xgb_model_spacy, train_features_tfidf_spacy, train_target, test_features_tfidf_spacy, test_target)


# ### Long Short-Term Memory
# 
# El código a continuación entrena y evalúa un modelo de Redes Neuronales Recurrentes (RNN) utilizando la arquitectura de Memoria a Largo Plazo (LSTM) con características generadas a partir de reseñas preprocesadas con NLTK. Comentarios sobre el código:
# 
# - Tokenización y padding: Este fragmento de código transforma las reseñas preprocesadas con NLTK en secuencias de tokens y luego las rellena o trunca a una longitud de 100 tokens.
# - Modelo LSTM: Este fragmento de código inicializa y configura un modelo LSTM. El modelo tiene una capa de embedding que transforma los tokens en vectores de 100 dimensiones, una capa LSTM con 100 unidades y una tasa de dropout del 20%, y una capa densa que realiza la clasificación final.
# - Entrenamiento del modelo: Este fragmento de código entrena el modelo LSTM en el conjunto de entrenamiento durante 5 épocas con un tamaño de lote de 64.
# - Evaluación del modelo: Este fragmento de código genera predicciones en los conjuntos de entrenamiento y prueba utilizando el modelo LSTM entrenado.

# In[18]:


# Tokenización y padding
tokenizer_nltk = Tokenizer(num_words=5000)
tokenizer_nltk.fit_on_texts(df_reviews_train['review_processed_nltk'])
X_train_nltk = tokenizer_nltk.texts_to_sequences(df_reviews_train['review_processed_nltk'])
X_test_nltk = tokenizer_nltk.texts_to_sequences(df_reviews_test['review_processed_nltk'])

X_train_nltk = pad_sequences(X_train_nltk, maxlen=100)  # Cambiado a 100
X_test_nltk = pad_sequences(X_test_nltk, maxlen=100)  # Cambiado a 100

# Modelo LSTM
lstm_model_nltk = Sequential()
lstm_model_nltk.add(Embedding(5000, 100, input_length=100))  # Cambiado a 100
lstm_model_nltk.add(SpatialDropout1D(0.2))
lstm_model_nltk.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
lstm_model_nltk.add(Dense(1, activation='sigmoid'))

lstm_model_nltk.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
lstm_model_nltk.fit(X_train_nltk, train_target, epochs=5, batch_size=64)

# Evaluación del modelo
train_pred_nltk = lstm_model_nltk.predict(X_train_nltk)
test_pred_nltk = lstm_model_nltk.predict(X_test_nltk)


# In[19]:


print("Train Accuracy with NLTK preprocessing:", accuracy_score(train_target, np.round(train_pred_nltk)))
print("Test Accuracy with NLTK preprocessing:", accuracy_score(test_target, np.round(test_pred_nltk)))


# In[20]:


print("Train F1 Score with NLTK preprocessing:", f1_score(train_target, np.round(train_pred_nltk)))
print("Test F1 Score with NLTK preprocessing:", f1_score(test_target, np.round(test_pred_nltk)))


# Los resultados muestran la evaluación de un modelo LSTM utilizando características generadas a partir de reseñas preprocesadas con NLTK. 
# 
# - Precisión: La precisión del modelo en el conjunto de entrenamiento es de 0.9741, lo que indica que el modelo clasifica correctamente el 97.41% de las reseñas en el conjunto de entrenamiento. Sin embargo, la precisión disminuye a 0.8574 en el conjunto de prueba, lo que sugiere que el modelo puede estar sobreajustando al conjunto de entrenamiento.
# - Valor F1: El valor F1 del modelo en el conjunto de entrenamiento es de 0.9740, lo que indica un buen equilibrio entre la precisión y el recall en el conjunto de entrenamiento. Sin embargo, al igual que la precisión, el valor F1 disminuye a 0.8558 en el conjunto de prueba.
# 
# Estos resultados sugieren que el modelo LSTM está sobreajustando al conjunto de entrenamiento, ya que su rendimiento es casi perfecto en el conjunto de entrenamiento pero disminuye en el conjunto de prueba. Podrías considerar técnicas para prevenir el sobreajuste, como agregar regularización, aumentar el dropout, o utilizar una validación cruzada más robusta para la selección del modelo.

# Redes Neuronales Recurrentes (RNN) utilizando la arquitectura de Memoria a Largo Plazo (LSTM) con características generadas a partir de reseñas preprocesadas con spaCy. 
# 
# - Tokenización y padding: Este fragmento de código transforma las reseñas preprocesadas con spaCy en secuencias de tokens y luego las rellena o trunca a una longitud de 100 tokens.
# - Modelo LSTM: Este fragmento de código inicializa y configura un modelo LSTM. El modelo tiene una capa de embedding que transforma los tokens en vectores de 100 dimensiones, una capa LSTM con 100 unidades y una tasa de dropout del 20%, y una capa densa que realiza la clasificación final.
# - Entrenamiento del modelo: Este fragmento de código entrena el modelo LSTM en el conjunto de entrenamiento durante 5 épocas con un tamaño de lote de 64.
# - Evaluación del modelo: Este fragmento de código genera predicciones en los conjuntos de entrenamiento y prueba utilizando el modelo LSTM entrenado.
# 
# En cuanto a las conclusiones generales, los resultados de entrenamiento muestran que el modelo LSTM mejora su rendimiento a lo largo de las épocas, alcanzando una precisión del 92.17% en la última época. 

# In[21]:


# Tokenización y padding
tokenizer_spacy = Tokenizer(num_words=5000)
tokenizer_spacy.fit_on_texts(df_reviews_train['review_processed_spacy'])
X_train_spacy = tokenizer_spacy.texts_to_sequences(df_reviews_train['review_processed_spacy'])
X_test_spacy = tokenizer_spacy.texts_to_sequences(df_reviews_test['review_processed_spacy'])

X_train_spacy = pad_sequences(X_train_spacy, maxlen=100)
X_test_spacy = pad_sequences(X_test_spacy, maxlen=100)

# Modelo LSTM
lstm_model_spacy = Sequential()
lstm_model_spacy.add(Embedding(5000, 100, input_length=X_train_spacy.shape[1]))
lstm_model_spacy.add(SpatialDropout1D(0.2))
lstm_model_spacy.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
lstm_model_spacy.add(Dense(1, activation='sigmoid'))

lstm_model_spacy.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
lstm_model_spacy.fit(X_train_spacy, train_target, epochs=5, batch_size=64)

# Evaluación del modelo
train_pred_spacy = lstm_model_spacy.predict(X_train_spacy)
test_pred_spacy = lstm_model_spacy.predict(X_test_spacy)


# In[22]:


print("Train Accuracy with spaCy preprocessing:", accuracy_score(train_target, np.round(train_pred_spacy)))
print("Test Accuracy with spaCy preprocessing:", accuracy_score(test_target, np.round(test_pred_spacy)))


# In[23]:


print("Train F1 Score with spaCy preprocessing:", f1_score(train_target, np.round(train_pred_spacy)))
print("Test F1 Score with spaCy preprocessing:", f1_score(test_target, np.round(test_pred_spacy)))


# - Precisión: La precisión del modelo en el conjunto de entrenamiento es de 0.9521, lo que indica que el modelo clasifica correctamente el 95.21% de las reseñas en el conjunto de entrenamiento. Sin embargo, la precisión disminuye a 0.8423 en el conjunto de prueba, lo que sugiere que el modelo puede estar sobreajustando al conjunto de entrenamiento.
# - Valor F1: El valor F1 del modelo en el conjunto de entrenamiento es de 0.9524, lo que indica un buen equilibrio entre la precisión y el recall en el conjunto de entrenamiento. Sin embargo, al igual que la precisión, el valor F1 disminuye a 0.8427 en el conjunto de prueba.
# 
# Estos resultados sugieren que el modelo LSTM está sobreajustando al conjunto de entrenamiento, ya que su rendimiento es casi perfecto en el conjunto de entrenamiento pero disminuye en el conjunto de prueba. Podrías considerar técnicas para prevenir el sobreajuste, como agregar regularización, aumentar el dropout, o utilizar una validación cruzada más robusta para la selección del modelo.

# ### Modelo BERT base acotada (CPU)

# In[21]:


# Dividir los datos originales en nuevos conjuntos con el 10% de los datos
df_reviews_train_h, _, train_target_h, _ = train_test_split(df_reviews_train, train_target, test_size=0.9, random_state=42)
df_reviews_test_h, _, test_target_h, _ = train_test_split(df_reviews_test, test_target, test_size=0.9, random_state=42)

# Verificar las formas de los nuevos conjuntos
print("Forma del nuevo conjunto de entrenamiento:", df_reviews_train_h.shape)
print("Forma del nuevo conjunto de prueba:", df_reviews_test_h.shape)


# In[22]:


# Descargar y cargar el modelo BERT
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')

# Definir función para obtener características BERT a partir de texto
def BERT_text_to_embeddings(texts, model, tokenizer, max_length=15, batch_size=5, disable_progress_bar=False):
    
    ids_list = []
    attention_mask_list = []

    # Tokenizar los textos y crear las máscaras de atención
    for text in texts:
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        ids_list.append(inputs['input_ids'])
        attention_mask_list.append(inputs['attention_mask'])

    # Define el dispositivo como 'cpu'
    device = torch.device('cpu')
        
    model.to(device)
    if not disable_progress_bar:
        print(f'Using device: {device}.')
        
    # Convertir las listas en tensores
    ids_tensor = torch.cat(ids_list, dim=0).to(device)
    attention_mask_tensor = torch.cat(attention_mask_list, dim=0).to(device)
    
    # obtener incrustaciones en lotes
    embeddings = []

    for i in tqdm(range(math.ceil(len(ids_list)/batch_size)), disable=disable_progress_bar):
        start = batch_size * i
        end = min(batch_size * (i + 1), len(ids_list))

        ids_batch = ids_tensor[start:end]
        attention_mask_batch = attention_mask_tensor[start:end]
            
        with torch.no_grad():            
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)   
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())
        
    return np.concatenate(embeddings)


# In[23]:


# Obtener características BERT para entrenamiento y prueba
train_features_bert_nltk = BERT_text_to_embeddings(df_reviews_train_h['review_processed_nltk'], model, tokenizer)
test_features_bert_nltk = BERT_text_to_embeddings(df_reviews_test_h['review_processed_nltk'], model, tokenizer)


# In[24]:


# Guardar las características en un archivo comprimido
np.savez_compressed('features_bert_h.npz', train_features_bert_nltk=train_features_bert_nltk, test_features_bert_nltk=test_features_bert_nltk)

# Cargar las características guardadas
with np.load('features_bert_h.npz') as data:
    train_features_bert_nltk = data['train_features_bert_nltk']
    test_features_bert_nltk = data['test_features_bert_nltk']

print("Dimensiones de características de entrenamiento:", train_features_bert_nltk.shape)
print("Dimensiones de características de prueba:", test_features_bert_nltk.shape)


# In[ ]:





# In[25]:


from sklearn.linear_model import LogisticRegression

# Asegurar que las etiquetas tengan el mismo número de muestras que las características
train_target_h = df_reviews_train_h['pos'][:len(train_features_bert_nltk)]
test_target_h = df_reviews_test_h['pos'][:len(test_features_bert_nltk)]

# Definir el modelo de Regresión Logística
logistic_regression_model = LogisticRegression()

# Entrenar el modelo con las características BERT de entrenamiento y las etiquetas de entrenamiento
logistic_regression_model.fit(train_features_bert_nltk, train_target_h)

# Predecir las etiquetas para las características BERT de prueba
predictions = logistic_regression_model.predict(test_features_bert_nltk)

# Calcular métricas de evaluación
accuracy = accuracy_score(test_target_h, predictions)
f1 = f1_score(test_target_h, predictions)
classification_rep = classification_report(test_target_h, predictions)
conf_matrix = confusion_matrix(test_target_h, predictions)

# Imprimir las métricas de evaluación
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)


# El código entrena y evalúa un modelo de Regresión Logística utilizando características BERT generadas a partir de reseñas preprocesadas. 
# 
# - Asegurar que las etiquetas tengan el mismo número de muestras que las características: Este fragmento de código asegura que los vectores de etiquetas tengan la misma longitud que los vectores de características. Esto es importante para evitar errores al entrenar el modelo.
# - Definir el modelo de Regresión Logística: Este fragmento de código inicializa un modelo de Regresión Logística.
# - Entrenamiento del modelo: Este fragmento de código entrena el modelo de Regresión Logística utilizando las características BERT del conjunto de entrenamiento y las etiquetas de entrenamiento.
# - Predicción de las etiquetas: Este fragmento de código utiliza el modelo entrenado para predecir las etiquetas para las características BERT del conjunto de prueba.
# - Cálculo de métricas de evaluación: Este fragmento de código calcula varias métricas de evaluación, incluyendo la precisión, el valor F1, el informe de clasificación y la matriz de confusión.
# 
# Los resultados muestran que el modelo de Regresión Logística tiene una precisión y un valor F1 de 0.6434 y 0.6557 respectivamente en el conjunto de prueba. Estas métricas indican cómo de bien se desempeña el modelo en la clasificación de las reseñas.

# ## Reseñas

# In[24]:


# Crear un DataFrame con las reseñas
my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

# Normalización
my_reviews['review_norm'] = my_reviews['review'].apply(clean_text)

# Preprocesamiento
my_reviews['review_processed'] = my_reviews['review_norm'].progress_apply(preprocess_text_nltk)

# Preprocesamiento con spaCy
my_reviews['review_spacy'] = my_reviews['review_norm'].progress_apply(preprocess_text_nltk)

# Mostrar el DataFrame con las reseñas preprocesadas
print(my_reviews)


# ### Modelo 2 : NLTK, TF-IDF y LR

# In[35]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = logistic_regression_model_nltk.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 3: spaCy, TF-IDF y LR

# In[36]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = logistic_regression_model_spacy.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 4: spaCy, TF-IDF y LGBMClassifier

# In[37]:


texts = my_reviews['review_norm']

my_reviews_pred_prob_lgbm1 = lgbm_classifier_spacy.predict_proba(tfidf_vectorizer.transform(texts.apply(lambda x: preprocess_text_nltk(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
print("\n")
my_reviews_pred_prob_lgbm2 = lgbm_classifier_spacy.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 5: NLTK, TF-IDF y LGBMClassifier

# In[38]:


texts = my_reviews['review_norm']

my_reviews_pred_prob_lgbm3 = lgbm_classifier_nltk.predict_proba(tfidf_vectorizer.transform(texts.apply(lambda x: preprocess_text_nltk(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')

print("\n")
my_reviews_pred_prob_lgbm4 = lgbm_classifier_nltk.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 6: SVM

# ### Modelo 7: Árbol de Decisión

# In[39]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = decision_tree_model.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
    
print("\n")

my_reviews_pred_prob = decision_tree_model_spacy.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 8: Bosque Aleatorio

# In[40]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = random_forest_model_nltk.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
    
print("\n")

my_reviews_pred_prob = random_forest_model_spacy.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 9: XGBoost

# In[ ]:


#texts = my_reviews['review_norm']

#my_reviews_pred_prob = xgb_model_nltk.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

#for i, review in enumerate(texts.str.slice(0, 100)):
#    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
    
#print("\n")

#my_reviews_pred_prob = xgb_model_spacy.predict_proba(tfidf_vectorizer.transform(texts))[:, 1]

#for i, review in enumerate(texts.str.slice(0, 100)):
#    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 10: Long Short-Term Memory

# In[31]:


texts = my_reviews['review_norm']

# Convert the sparse matrix to a dense matrix
texts_transformed = tfidf_vectorizer2.transform(texts).todense()

# Now you can use the predict method
my_reviews_pred_prob = lstm_model_nltk.predict(texts_transformed)

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i][0]:.2f}:  {review}')
    
print("\n")

# Assuming lstm_model_spacy is another model you have
my_reviews_pred_prob = lstm_model_spacy.predict(texts_transformed)

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i][0]:.2f}:  {review}')


# ### Modelo 11: BERT

# In[41]:


# Entrenar el modelo de Regresión Logística con las características BERT
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(train_features_bert_nltk, train_target_h)

# Entrenar el modelo LGBM con las características TF-IDF
lgbm_classifier_spacy = LGBMClassifier()  # Asegúrate de que este es el modelo correcto
lgbm_classifier_spacy.fit(tfidf_vectorizer.transform(df_reviews_train_h['review_processed_nltk']), train_target_h)

# Obtener características BERT para las reseñas
texts = my_reviews['review_norm']
my_reviews_features_bert = BERT_text_to_embeddings(texts, model, tokenizer, disable_progress_bar=True)

# Obtener características TF-IDF para las reseñas
my_reviews_features_tfidf = tfidf_vectorizer.transform(texts.apply(lambda x: preprocess_text_nltk(x)))

# Predicciones con el modelo de Regresión Logística
my_reviews_pred_prob_bert = logistic_regression_model.predict_proba(my_reviews_features_bert)[:, 1]

# Predicciones con el modelo LGBM
my_reviews_pred_prob_tfidf = lgbm_classifier_spacy.predict_proba(my_reviews_features_tfidf)[:, 1]

# Mostrar las probabilidades de predicción para cada reseña
for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'BERT: {my_reviews_pred_prob_bert[i]:.2f}, TF-IDF: {my_reviews_pred_prob_tfidf[i]:.2f}:  {review}')


# ## Conclusiones

# Según los resultados obtenidos de los dos modelos con la metrica F1 mayor:
# 
# - El modelo de Regresión Logística con preprocesamiento NLTK tiene un valor F1 de 0.87 en el conjunto de entrenamiento y 0.86 en el conjunto de prueba.
# - El modelo LGBMClassifier con preprocesamiento spaCy tiene un valor F1 de 0.91 en el conjunto de entrenamiento y 0.85 en el conjunto de prueba.
# Ambos modelos muestran un buen rendimiento, pero si nos basamos en la métrica F1 en el conjunto de prueba, el modelo de Regresión Logística con preprocesamiento NLTK es ligeramente superior con un valor F1 de 0.86.
# 
# Es importante mencionar que aunque el modelo de Regresión Logística tiene un mejor rendimiento en el conjunto de prueba según la métrica F1, el modelo LGBMClassifier tiene un mejor rendimiento en el conjunto de entrenamiento. Esto podría indicar que el modelo LGBMClassifier está sobreajustando al conjunto de entrenamiento. Podríamos considerar técnicas para prevenir el sobreajuste, como la regularización, para mejorar el rendimiento del modelo LGBMClassifier en el conjunto de prueba.

# In[ ]:




