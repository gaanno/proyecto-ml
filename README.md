# INTEGRANTES
Diego Jiménez Muñoz
Gabriel Norambuena Muñoz

# INSTRUCCIONES DOCKER
docker build -t proyecto-ml-dj-gn .
docker run -ti proyecto-ml-dj-gn /bin/bash

# PREPROCESAMIENTO
python3 preprocesamiento.py

# ENTRENAMIENTO
python3 entrenamiento.py

# PREDICCION
python3 prediccion.py