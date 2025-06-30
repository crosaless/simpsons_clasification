# Proyecto para detectar Personajes de Los Simpsons utilizando pérdida de las trillizas

Este proyecto implementa un sistema de reconocimiento de personajes de Los Simpsons mediante técnicas de aprendizaje profundo y aprendizaje de representaciones. Utiliza una red neuronal entrenada con pérdida de la trilliza (triplet loss), una estrategia que permite aprender un espacio de embedding donde imágenes del mismo personaje estén cercanas y de diferentes personajes estén alejadas.

El modelo fue entrenado sobre un conjunto de imágenes etiquetadas de los personajes más icónicos, y luego se utiliza para identificarlos a partir de su representación vectorial.

El modelo elegido fue una densenet121 por los resultados obtenidos en los accuracy de entrenamiento y testeo.  

## Estructura del proyecto
```
    proyecto/
├── prod/
│   ├── app.py                   #Aplicación principal
│   ├── utils.py                 #Funciones auxiliares
│   ├── modelo.pth               #Modelo entrenado
│   ├── reference_embeddings.pt  #Embeddings de referencia de cada clase
│   ├── avatares/                #Carpeta de avatares
│   │   ├── homer_simpson.png
│   │   ├── bart_simpson.jpg
│   │   └── ... (más avatares)
│   └── requirements.txt
├── data/   #contiene el dataset usado en el entrenamiento y los archivos auxiliares de prepocesamiento
│── dev/    #notebooks realizados en colab para entrenar los modelos
└── README.md 
```


## Clonar repositorio

    git clone https://github.com/crosaless/simpsons_clasification.git


## Instalar Dependencias

    pip install -r prod/requirements.txt

## Ejecutar aplicación

    streamlit run prod/app.py

O entra a la aplicación en Community Cloud:

[Cliquea aqui](https://simpsonsclasification.streamlit.app/)
