>## [Predicción de rating de juegos de mesa.](https://github.com/Kuja182/Prediccion-rating-de-usuarios-de-juegos-de-mesa)
En este proyecto de Machine Learning vamos a predecir como va a ser el rating de los usuarios en función de las caracteristicas de los juegos y como mostrarselo a startups que quiera hacer un juego de mesa.  
Todos los juegos que aparecen en este dataset de juegos de mesa son los que han sido valorados al menos 30 veces por usuarios.    
Para hacer la predicción seguiremos el siguiente orden:  
> ### 1. [Preprocesamiento de datos y creación de nuevas features para predicción.](https://github.com/Kuja182/Prediccion-sobre-juegos-de-mesa/blob/main/notebooks/01_Preprocesamiento.ipynb)  
- Obtendremos dos datasets con las siguientes caracteristicas:  
  1 Tener una idea general del problema y analizarlo sin crear nuevas variables.
  2 Crear nuevas variables y transformarlas en variables utiles para hacer nuestro modelo de ML.  
> ### 2. [EDA (Análisis de datos de los datasets procesados)](https://github.com/Kuja182/Prediccion-sobre-juegos-de-mesa/blob/main/notebooks/02_EDA.ipynb)    
- Sabremos cuales son las variables que mayor relación tienen con nuestro problema a valorar.  
> ### 3. [Construcción y entrenamiento de modelos de Machine Learning.](https://github.com/Kuja182/Prediccion-sobre-juegos-de-mesa/blob/main/notebooks/03_Entrenamiento_Modelo.ipynb)       
- Elaboraremos y entrenaremos respecto a variables importantes de juegos para ver su predicción y después entrenarlas.  
> ### 4. [Evaluación de los modelos predictivos.](https://github.com/Kuja182/Prediccion-sobre-juegos-de-mesa/blob/main/notebooks/04_Evaluacion_Modelo.ipynb)    
- Con métricas concretas evaluaremos nuestros modelos previamente entrenados.  
> ### 5. [App para el cliente final.](https://github.com/Kuja182/Prediccion-sobre-juegos-de-mesa/tree/main/app)  
- Con streamlit lanzaremos un aplicación local donde predecir que rating de usuarios tenemos ante ciertos parametros de un juego de mesa nuevo.
## OBTENCIÓN DE LOS DATOS
### DATASETS Y FUENTES ALTERNATIVAS DE DATOS
https://www.kaggle.com/datasets/andrewmvd/board-games
Dilini Samarasinghe, July 5, 2021, "BoardGameGeek Dataset on Board Games", IEEE Dataport, doi: https://dx.doi.org/10.21227/9g61-bs59.
