from yolo_impl.YoloImplementation import YoloImplementation
import time

# Existe el output del modelo 1814.weights precargado en la carpeta /output con el nombre model_output.csv y su respectivo excel.
# Si se quiere probar o el 1502.weights, cambiar el texto del parametro partial de la linea 15 por '1502.weights'.

# En la carpeta yolo_impl se encuentra el archivo .py conteniendo la clase que implementa los modelos SDF y DOCR entrenados de Yolo con transfer learning.
# En la carpeta test_images se encuentran las imagenes de prueba
# En la carpeta detected_dates están los crops de las fechas detectadas. Se pueden borrar y volver a correr el modelo si gustan, no hay problema.
# En la carpeta output se cargan las salidas del modelo
# En la carpeta yolov3_models existen 2 carpetas: docr y sdf. En ambas estan las configuraciones de yolo con los pesos optimizados respectivos para
# ejecutarse.
# En la carpeta training_data estan los datos de entrenamiento y un archivo 'training.txt' donde se da creditos al tutorial que se siguió para
# realizar el transfer learning con Yolov3.
# En la carpeta sample_detections están 2 muestras de cómo detectan los modelos sus respectivas tareas.

def main():
    start_time = time.time()

    print("Iniciando detección de firmas y campos de fecha con el modelo Yolo-SDF...")
    yimp = YoloImplementation()
    yimp.excecute_sdf()
    print("Yolo-SDF ha terminado exitosamente.")
    print("Iniciando detección de números y formateando las fechas con el modelo Yolo-DOCR...")
    yimp.execute_docr(partial='1814.weights') # Cambiar por '1502.weights' si se quiere ver los resultados del otro modelo
    print("Yolo-DOCR ha terminado exitosamente.")
    print("El modelo ha corrido exitosamente. Dirígete a la carpeta /output para ver las salidas del modelo.")
    print("Se realizó la detección de firmas y fechas de las 107 imágenes de prueba en {} segundos".format(time.time() - start_time))

if __name__ == "__main__":
    main()