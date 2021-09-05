# TFG de Ander Gil Moro: Diccionarios Inversos
Repositorio que almacena los scripts y predicciones de los prototipos creados.

En la carpeta "Scripts" se almacenan los scripts usados durante el proyecto.

En la carpeta "Predicciones", se almacenan las predicciones realizadas por los tres prototipos en las pruebas estandarizadas. 
-En el caso de Mask-Filling, se muestran las palabras clave y las respectivas 100 primeras palabras predecidas. También se han añadido los resultados de las diferentes pruebas de fine-tuning.
-En el caso de Sentence-Transformers y la API de Datamuse, se muestran las palabras clave y las primeras predicciones hasta encontrar la palabra objetivo, o en caso de que no se encuentre, las primeras 100 predicciones.

En la carpeta "Data" se almacenan los conjuntos de datos estandarizados usados para la evaluación de los modelos
-Seen (Definiciones vistas)
-Unseen (Definiciones no vistas durante el entrenamiento)
-Desc (Descripciones dadas por personas)
También se han añadido los documentos de WordNet usados para el entrenamiento y evaluación del primer prototipo, Mask-Filling:
-Los que contienen 100 definiciones, que han sido usados para encontrar los mejores hiperparámetros del fine-tuning.
-Los que contienen el diccionario entero, exceptuando las palabras claves del dataset de evaluación "Unseen".
