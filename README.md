
# KichAI Team - Entel Dataton

En este documento se detallará una breve descripción del modelo implementado (se extiende en los comentarios del código) y una descripción de los requerimientos y pasos para ejecutar el proyecto.




## YOLOv3-SDF y YOLOv3-DOCR

El modelo implementado está basado en dos modelos construidos y entrenados con transfer learning a partir de YOLOv3 (Modelo estado del arte en Object Detection and Recognition).

### YOLOv3-SDF

Es un modelo derivado de YOLOv3 con transfer learning encargado de detectar fechas y firmas a partir del input de una imagen.

Finalmente termina recortando la fecha detectada para ser procesada por el YOLOv3-DOCR.

### YOLOv3-DOCR

Es un modelo derivado de YOLOv3 con transfer learning encargado de detectar y clasificar los números dentro de las fechas recortadas detectadas por el modelo YOLOv3-SDF.

A partir de estos números se aplican una serie de algoritmos para separarlos efectivamente por dia, mes y año.

### Entrenamiento

El entrenamiento fue realizado en un entorno de Google Colab y se etiquetaron las imágenes de entrenamiento utilizando la herramienta labelimg obtenida de github. Más detalle se puede observar en la carpeta /training_data.

Se eligieron los pesos de las épocas con mejor performance. 

### Muestras

Una muestra de las detecciones hechas por nuestros modelos se puede encontrar en la carpeta /sample_detections.

## Ejecución del proyecto

El código está completamente comentado. ```app.py``` en la raíz del proyecto ejecuta el código. Y ```YoloImplementation.py``` en la carpeta ```/yolo_impl``` ejecuta la implementación y algoritmos utilizados en todo el modelo.

### Dependencias

- Librería time (viene con python)
- Librería  os (viene con python)
- Librería opencv ```pip install opencv-python==4.5.2```
- Librería numpy ```pip install numpy```
- Librería pandas ```pip install pandas```

### Ejecución

- Clonar repositorio de git ```git clone https://github.com/LuisRivera1699/KichAI-EntelDataton.git```
- Descargar la carpeta yolov3_models. Descomprimir en la raíz. (En esta carpeta están los pesos y los modelos entrenados. Y github no nos permite subir más de 100mb). Link de descarga: (https://drive.google.com/drive/folders/1iSDbVLtyADoSm0WWYhiTQ1ppUC9c9te6?usp=sharing)
- Dirigirse a la raíz del proyecto y ejecutar: ```app.py``` o ```python app.py```
- Esperar a que el modelo procese las imágenes. Demora aproximadamente 80 segundos.
- Cuando el modelo haya acabado, revisar la carpeta /output. Dentro de ella se encontrará un .csv y un .xlsx con el nombre 'model_output' contenido las salidas del modelo.

## Contacto

Cualquier consulta o pregunta en la etapa de evaluación, por favor contactarnos a (contesta siempre):

- Email: luisriveradiaz1699@gmail.com
- Celular: +51 932 104 502
## Autores

Luis Rivera
- Email: luisriveradiaz1699@gmail.com

Pedro Zúñiga
- Email: pedro.zuniga9749@gmail.com

Iver Castro
- Email: ivrcstrrvr@gmail.com

  
## Referencias

Se agradece y dan créditos por la ayuda que nos brindaron a las siguientes guías y repositorios open source:

 - [YOLOv3 Custom Object Detection with Transfer Learning](https://medium.com/analytics-vidhya/yolov3-custom-object-detection-with-transfer-learning-47186c8f166d). Guía de Transfer Learning con YOLOv3.
 - [LabelImg](https://github.com/tzutalin/labelImg). Herramienta de etiquetado en formato YOLOv3.
 - [Darknet Architecture](https://github.com/AlexeyAB/darknet). Arquitectura de red sobre la cual corre YOLOv3.
 - [YOLOv3 Weights](https://pjreddie.com/media/files/darknet53.conv.74). Pesos pre-entrenados de YOLOv3.
