# Integrantes
- Diego Jiménez Muñoz
- Gabriel Norambuena Muñoz

# Instrucciones Docker
Construir Contenedor Docker

 ```bash
docker build -t proyecto-ml-dj-gn .
 ```

Ejecutar Contenedor
 ```bash
docker run -ti proyecto-ml-dj-gn /bin/bash
 ```

# Preprocesamiento
 ```bash
python3 preprocesamiento.py
 ```

# Entrenamiento
Los modelos se encuentran en la carpeta [models](models/), ejecutar el siguiente comando en caso de querer entrenar nuevamente.
 ```bash
python3 entrenamiento.py
 ```

# Predicción
 ```bash
python3 prediccion.py
 ```
