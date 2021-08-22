import cv2
import numpy as np
import os
import pandas as pd

# YoloImplementation es la clase que implementa la detección de objetos de los dos modelos
# de Yolov3 (You only look once model) tuneados mediante aprendizaje transferido.

# Consta de dos modelos: SDF y DOCR, implementados por excecute_sdf y excecute_docr respectivamente
# SDF fue llamado por sus siglas Signature and Date Fields, debido a que este modelo es el encargado de
# detectar las firmas en el documento y el campo de fecha, en caso haya una fecha escrita. Recibe las imagenes
# completas, detecta las firmas y el campo de fecha, corta el campo de fecha detectado y lo guarda en la carpeta detected_dates.
# DOCR fue llamado por sus siglas Date Optical Character Recognition, debido a que este  modelo es el 
# encargado de detectar los numeros dentro de las fechas cortadas por el SDF. Internamente aplica un algoritmo de 
# formateo de fechas para dividir la lista de números detectados en un formato de fecha. Recibe las imagenes
# cortadas de las fechas por el SDF, formatea los numeros detectados y guarda la salida en model_output.csv.

# Los datos utilizados para hacer el transfer learning de Yolov3 fueron los proveidos por el concurso. Y fueron etiquetados
# en formato Yolo utilizando la herramienta labelimg obtenida de gitlab https://github.com/tzutalin/labelImg.
# La darknet, arquitectura de redes neuronales, donde corre yolo fue obtenida de https://github.com/AlexeyAB/darknet.
# Y los pesos preentrenados de yolo fueron obtenidos de https://pjreddie.com/media/files/darknet53.conv.74

# El entrenamiento fue hecho en un entorno de Google Colab, con una tarjeta gráfica P100-PCIE de 16GB.
# El entrenamiento de Yolov3-SDF consto de 2 clases (signature y date_field) y una configuracion de 4000 max_batches y 21 filtros. (4000 epochs)
# El tiempo de entrenamiento fue de 3 horas con 11 minutos.
# Se eligio los pesos de la ultima epoca de entrenamiento debido su perfecto funcionamiento 100%.
# Para la implementacion se utiliza lo siguiente:
# /yolov3_models/sdf/classes.txt -> archivo de texto conteniendo las clases del modelo
# /yolov3_models/sdf/yolov3_testing.cfg -> arquitectura de la red neuronal a cargar
# /yolov3_models/sdf/yolov3_training_last.weights -> archivo con los pesos entrenados en el transfer learning en la ultima epoca. (mejor resultado)

# Y el entrenamiento de Yolov3-DOCR consto de 10 clases (del 0 al 9) y una configuracion de 20000 max_batches y 45 filtros. (20000 epochs)
# El tiempo de entrenamiento fue de 20 horas aproximadamente.
# Se eligio los pesos de la epoca guardada a las 18:14pm del dia 18 de agosto debido a su alto performance. (no detalla el numero de epoca)
# /yolov3_models/docr/classes.txt -> archivo de texto conteniendo las clases del modelo
# /yolov3_models/docr/yolov3_testing.cfg -> arquitectura de la red neuronal a cargar
# /yolov3_models/docr/partial/1814.weights -> archivo con los pesos entrenados en el transfer learning en la epoca de las 18:14pm. (segundo mejor resultado)
# /yolov3_models/docr/partial/1502.weights -> archivo con los pesos entrenados en el transfer learning en la epoca de las 15:02pm. (mejor resultado)
class YoloImplementation:
    def __init__(self):
        pass

    # set_list es una funcion que recibe una lista y le quita los numeros repetidos.
    def set_list(self, laux):
        aux = []
        for item in laux:
            if item not in aux:
                aux.append(item)
        return aux

    # excecute_sdf es una funcion que implementa el modelo Yolov3-SDF
    # Bloques de formateo de salidas explicadas con comentarios dentro de la función.
    def excecute_sdf(self):
        # Se cargan los pesos y la arquitectura de redes de la carpera /yolov3_models/sdf/yolov3_training_last.weights
        net = cv2.dnn.readNet('./yolov3_models/sdf/yolov3_training_last.weights', 
            './yolov3_models/sdf/yolov3_testing.cfg')

        classes = []
        with open("./yolov3_models/sdf/classes.txt", "r") as f:
            classes = f.read().splitlines()

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(100, 3))

        # Leemos las imagenes del directorio /test_images
        images = os.listdir('./test_images')

        # Creamos una lista donde se guardaran las predicciones. Esta lista luego se convertira en
        # dataframe para poder ser exportado a un csv con los datos de esta primer modelo formateados.
        predictions = []

        # Se recorre la lista de imagenes
        for i in images:
            image_name = i

            # Diccionario que guardará las salidas de cada imagen procesada por el SDF.
            img_dict = dict(
                id=image_name.split('.')[0],
                sign_1=0,
                sign_2=0,
                date_day=0,
                date_month=0,
                date_year=0
            )

            # Lectura de imagen
            img = cv2.imread('./test_images/{}'.format(image_name))
            height, width, _ = img.shape

            # Preprocesamiento de imagen, ingreso al modelo y obtencion de salida del podelo en layerOutputs
            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            # Lista para guardar las salidas del modelo: boxes -> recuadros de deteccion, confidences -> confianza de la salidas
            # class_ids -> clases detectadas
            boxes = []
            confidences = []
            class_ids = []

            # Lista para guardar las firmas detectadas
            detected_signatures = []

            for output in layerOutputs:
                for detection in output:

                    # Extrae las salidas
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Solamente procesa las que tienen confianza mayor a 0.2 (mejores practicas de Yolov3)
                    if confidence > 0.2:

                        # Puntos y medidas del objeto detectado
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

                        # Si la clase detectada es 1 (date_field) se asume que el año es 2021 porque sí existe la fecha.
                        # No es determinante, en el DOCR se determina el formato del año.
                        if class_id == 1:
                            img_dict["date_year"] = 2021

                        # Si la clase detectada es 0 (signature) se procesan los resultados de las firmas.
                        elif class_id == 0:
                            # Si aun no se ha detectado una firma, se agregan los centros a la lista de firmas detectdas
                            # para hacer validaciones con las posteriores firmas. Esto debido a que en algunos casos
                            # el modelo detecta dos firmas superpuestas, que en realidas son una sola.
                            if len(detected_signatures) == 0:
                                detected_signatures.append((center_x, center_y))
                            else:
                                # Variable para determinar si se debe agregar o no la firma a la lista.
                                append = False
                                # Ser recorre la lista
                                for i in detected_signatures:
                                    # Si el centro de la firma actualmente detectada coincide con el centro x con una precision
                                    # de +- 10 se valida lo mismo pero con el centro y para ver que no esten en la misma posicion x
                                    # pero diferente posicion y (ejemplo, uno arriba y otro abajo de la pagina).
                                    # Si estan en el rango de ambas, no se agregan, si no, se agrega la fecha a la lista.
                                    if center_x < i[0] + 10 and center_x > i[0] - 10:
                                        if center_y < i[1] + 10 and center_y > i[1] - 10:
                                            append = False
                                            break
                                        else:
                                            append = True
                                    else:
                                        append = True
                                if append:
                                    detected_signatures.append((center_x, center_y))
            
            # Debido a que la deteccion es perfecta
            # Si la lista de firmas tiene 1 sola firma, se determina su posición de acuerdo
            # a si su esquina derecha superior ha pasado la mitad de la imagen.
            # Si la lista de firmas tiene 2 firmas, se deduce que existen las dos firmas.
            if len(detected_signatures) == 1:
                if x+w < (width/2):
                    img_dict["sign_1"] = 1
                else:
                    img_dict["sign_2"] = 1
            elif len(detected_signatures) == 2:
                img_dict["sign_1"] = 1
                img_dict["sign_2"] = 1

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            # Recorre las imagenes detectadas para hacer un recorte de los campos de fecha detectados
            # Si existe, recorta el campo de fecha con un extra de +-3 a lo largo y a lo alto y lo guarda
            # en la carpeta de /detected_dates para ser analizado por el DOCR.
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    if class_ids[i] == 1:
                        cropped = img[y-3:y+h+3, x-3:x+w+3]
                        if y > (height/2):
                            cropped = cv2.rotate(cropped, cv2.ROTATE_180)
                        cv2.imwrite('./detected_dates/{}'.format(image_name), cropped)

            # Imprime el resultado de salida
            print(img_dict)
            # Agrega el resultado de salida a la lista de predicciones
            predictions.append(img_dict)

        # Se convierte la lista predictions a un dataframe de pandas y se exporta el dataframe a csv.
        sdf_output = pd.DataFrame(predictions)
        sdf_output.to_csv('./output/model_output.csv', index=False)

    # excecute_docr es una funcion que implementa el modelo Yolov3-DOCR
    # Bloques de formateo de salidas explicadas con comentarios dentro de la función.
    def execute_docr(self, partial):
        # Se cargan los pesos y la arquitectura de redes de la carpera /yolov3_models/docr/yolov3_{partial}.weights
        net = cv2.dnn.readNet('./yolov3_models/docr/partial/{}'.format(partial), 
            './yolov3_models/docr/yolov3_testing.cfg')

        classes = []
        with open("./yolov3_models/docr/classes.txt", "r") as f:
            classes = f.read().splitlines()

        # Se lee el directorio de /detected_dates
        images = os.listdir('./detected_dates')

        # Se carga el csv model_output de la carpeta /output que contiene las salidas del SDF corrido previamente.
        model_output = pd.read_csv('./output/model_output.csv')

        # Se recorren las imagenes
        for i in images:
            image_name = i

            # Lectura de imagenes
            img = cv2.imread('./detected_dates/{}'.format(image_name))
            height, width, _ = img.shape

            # Preprocesamiento de imagen, ingreso al modelo y obtencion de salida del podelo en layerOutputs
            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            # Listas de salidas de cajas, confianzas y clases detectadas
            boxes = []
            confidences = []
            class_ids = []

            # Se iteran las salidas y se procesan solo las de confianza mayor a 0.2 (buenas practicas YOLO)
            for output in layerOutputs:
                for detection in output:

                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.2:

                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            # Se setea la lista de objetos detectados en formato de caja
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            # Se setean las listas para guardar el numero detectado, el x,y y ancho de la caja
            # del numero detectado
            n_list = []
            X_list = []
            y_list = []
            w_list = []

            # Se recorren las salidas detectadas si es que hay salidas detectadas
            if len(indexes)>0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])

                    # Si x es mayor a -1 (esta en la figura) y y es mayor a 2 (esta al menos 2 pixeles)
                    # abajo del tope (por el recorte con el que se guardo la imagen) se consideran las detecciones.
                    if x>-1 and y>2:
                        n_list.append(int(label))
                        X_list.append(x)
                        y_list.append(y)
                        w_list.append(w)

            # Se crea la lista de fechas ordenadas y una variable de stop para controlar el proximo while loop.
            ordered_date = []
            stop = False

            # Se itera la lista de posiciones xs de las cajas de numeros detectados y se va agregando
            # el menor a la lista de ordered_date. La logica es para ordenar los numeros de izquierda a derecha.
            while not stop:
                aux = min(X_list)
                idx = X_list.index(aux)
                ordered_date.append((n_list[idx], aux, y_list[idx], w_list[idx]))
                X_list.pop(idx)
                n_list.pop(idx)
                y_list.pop(idx)
                w_list.pop(idx)
                if len(X_list) == 0:
                    stop = True

            # Se calculan las separaciones entre las xs de los numeros consecutivos y se guardan en una lista.
            # Esto para la logica de formateo de fechas.
            separations = []
            for i in range(len(ordered_date)):
                if i != 0:
                    separations.append(ordered_date[i][1] - ordered_date[i-1][1])


            # Se crea una lista de listas de fecha que contendra en la primera posicion el dia, en la segunda el mes
            # y en la tercera el año.
            # Las variables day_f y month_f son para determinar que ya se termino de completar el dia y el mes.
            date = [[], [], []]
            day_f = False
            month_f = False

            # Logica de formateo de fechas:
            # Se revisaron los resultados obtenidos de los datos de entrenamiento y mediante un analisis de cada uno de estos
            # se consideraron las siguientes validaciones para detectar el los numeros que corresponden al dia, al mes y al año.

            # Si son 10 numeros detectados aceptados y los numeros en la posicion 2 y 5 (indexados desde 0) son 1,
            # es muy probable que el 1 detectado sea un slash (/) separador de fecha, por lo cual se obvian se 
            # formatea la fecha con los dos primeros numeros en el dia, los 2 segundos en el mes y los 4 ultimos en el año.
            if len(ordered_date)==10 and ordered_date[2][0] == 1 and ordered_date[5][0] == 1:
                date[0] = [n[0] for n in ordered_date[0:2]]
                date[1] = [n[0] for n in ordered_date[3:5]]
                date[2] = [n[0] for n in ordered_date[6:]]
            # Si existe mas de un numero detectado (por ello hay separaciones en la lista de separaciones) y
            # la distancia entre el primer numero y el ultimo es de mas de 45 pixeles, es muy proble que se haya
            # detectado solo los numeros del dia y del año y obviado los del mes. Por lo tanto la separacion es por dia
            # y por año.
            # Esta validacion se hizo para el primer modelo de DOCR entrenado con solo 61 imagenes de entrenamiento donde no lograba
            # captar todos los numeros. Actualmente ya se puede dejar de usar. Pero se deja como historico de validacion del score en kaggle.
            elif len(separations) > 0 and separations[0] > 45:
                date[0].append(ordered_date[0][0])
                date[2] = [n[0] for n in ordered_date[1:]]
            # Si la cantidad de numeros detectados es menor o igual a 3, es porque el modelo no ha detectado la mayoria de numeros
            # de la fecha. Entonces se opta por dividir la imagen en 3 secciones horizontalmente. Los numeros que se encuentran en la primera
            # seccion corresponden a dias, los de la sgunda a meses y los de la tercera a años.

            # Esta validacion se hizo para el primer modelo de DOCR entrenado con solo 61 imagenes de entrenamiento donde no lograba
            # captar todos los numeros. Actualmente ya se puede dejar de usar. Pero se deja como historico de validacion del score en kaggle.
            elif len(ordered_date) <= 3:
                partial_width = width/3
                for i in ordered_date:
                    if i[1]+(i[3]/2) < partial_width-4:
                        date[0].append(i[0])
                    elif i[1]+(i[3]/2) < (partial_width*2)-4:
                        date[1].append(i[0])
                    else:
                        date[2].append(i[0])
            # Si no cumple con estas validaciones entonces se pasa al formateo estandar que siguen la mayoria de imagenes detectadas.
            else:
                # Se observo que la cantidad de fechas detectadas con una longitud de 65 pixeles como maximo, tiene una separacion minima de
                # 15 pixeles entre el dia y el mes, y una separacion minima de 8 pixeles entre el mes y el año.
                # Las separaciones se cuentan desde el primer numero del elemento y el primer numero del siguiente elemento.
                if width <= 65:
                    count = 0
                    date[0].append(ordered_date[0][0])
                    for i in range(len(separations)):
                        count += separations[i]

                        if not day_f:
                            if count < 15:
                                date[0].append(ordered_date[i+1][0])
                            else:
                                day_f = True
                                count = 0
                        
                        if not month_f and day_f:
                            if count < 8:
                                date[1].append(ordered_date[i+1][0])
                            else:
                                month_f = True
                                count = 0
                        if month_f:
                            date[2] = [n[0] for n in ordered_date[i+1:]]
                            break
                # Se observo que la cantidad de fechas detectadas con una longitud de 125 pixeles como maximo, tiene una separacion minima de
                # 9 pixeles entre el dia y el mes, y una separacion minima de 11 pixeles entre el mes y el año.
                # Para validar los casos donde no se detectan los numeros del dia, se observo que las fechas detectadas con esta misma longitud
                # sus posiciones del ultimo numero del dia eran como maximo de 25 pixeles, por lo que cualquier numero detectado despues de 
                # esa posicion corresponde al mes.
                # Las separaciones se cuentan desde el primer numero del elemento y el primer numero del siguiente elemento.
                elif width <= 125:
                    count = 0
                    if ordered_date[0][1] > 25:
                        date[1].append(ordered_date[0][0])
                        day_f = True
                    else:
                        date[0].append(ordered_date[0][0])

                    for i in range(len(separations)):
                        count += separations[i]

                        if not day_f:
                            if count < 9:
                                date[0].append(ordered_date[i+1][0])
                            else:
                                day_f = True
                                count = 0
                        
                        if not month_f and day_f:
                            if count < 11:
                                date[1].append(ordered_date[i+1][0])
                            else:
                                month_f = True
                                count = 0
                        if month_f:
                            date[2] = [n[0] for n in ordered_date[i+1:]]
                            break
                # Se observo que la cantidad de fechas detectadas con una longitud de 126 pixeles como minimo, tiene una separacion minima de
                # 20 pixeles entre el dia y el mes, y una separacion minima de 24 pixeles entre el mes y el año.
                # Las separaciones se cuentan desde el primer numero del elemento y el primer numero del siguiente elemento.
                else:
                    count = 0
                    date[0].append(ordered_date[0][0])
                    for i in range(len(separations)):
                        count += separations[i]

                        if not day_f:
                            if count < 20:
                                date[0].append(ordered_date[i+1][0])
                            else:
                                day_f = True
                                count = 0
                        
                        if not month_f and day_f:
                            if count < 24:
                                date[1].append(ordered_date[i+1][0])
                            else:
                                month_f = True
                                count = 0

                        if month_f:
                            date[2] = [n[0] for n in ordered_date[i+1:]]
                            break

            # Se crea el diccionario que guardara el modelo de la fecha formateada
            date_dict = dict(
                id=image_name.split('.')[0],
                date_day=0,
                date_month=0,
                date_year=0
            )

            # Para cada uno de los elementos de la fecha (dia=0, mes=1, año=2)
            for i in range(len(date)):
                # Si es el dia
                if i == 0:
                    l = len(date[i])
                    # Si la longitud de la lista de dia es 1, entonces se agrega el numero como dia
                    if l == 1:
                        date_dict['date_day'] = date[i][0]
                    # Si la longitud de la lista de dia es 2
                    elif l == 2:
                        # Si el primer numero es mayor a 3, ya no es logico que tenga un numero despues, entonces
                        # se toma directamente ese numero como dia.
                        if date[i][0] > 3:
                            date_dict['date_day'] = str(date[i][0])
                        # Si el primer numero es 3 y el segundo es mayor a 1, no es logico, entonces se toma
                        # 3 como el dia y se elimina el segundo
                        elif date[i][0] == 3 and date[i][1] > 1:
                            date_dict['date_day'] = str(date[i][0])
                        # En todos los otros casos, se agregan los dos numeros en orden como dia
                        else:
                            date_dict['date_day'] = str(str(date[i][0])+str(date[i][1]))
                    # Si la longitud de la lista de dia es mayor a 2 (pasa cuando se detectan superpuestos o errores)
                    # Se utiliza la funcion set_list para eliminar repetidos.
                    elif l>2:
                        list_set = self.set_list(date[i])
                        # Si la longitud del la lista sin repetidos es de 1
                        # es muy probable que haya habido una trilogia de numeros repetidos (ejemplo 222) y muy
                        # poco probable que se haya detectado en la misma posicion por el performance observado del modelo
                        # por lo cual, se agrega un duplicado de ese numero.
                        if len(list_set) == 1:
                            date_dict['date_day'] = str(str(list_set[0])+str(list_set[0]))
                        # Si no, se agregan los primeros dos numeros.
                        else:
                            date_dict['date_day'] = str(str(list_set[0])+str(list_set[1]))
                # Si es el mes
                if i == 1:
                    l = len(date[i])
                    # Si la cantidad de numeros detectados en el mes es de 1, entonces se agrega ese numero como mes.
                    if l == 1:
                        date_dict['date_month'] = date[i][0]
                    # Si la cantidad de numeros detectados en el mes es de 2
                    elif l == 2:
                        # Como hay ocasiones donde ponen el dia en el campo de mes, entonces se utiliza la misma validacion
                        # que se hace en el campo de dias.
                        if date[i][0] > 3:
                            date_dict['date_month'] = str(date[i][0])
                        elif date[i][0] == 3 and date[i][1] > 1:
                            date_dict['date_month'] = str(date[i][0])
                        else:
                            date_dict['date_month'] = str(str(date[i][0])+str(date[i][1]))
                    # Si la cantidad de numeros detectados en el mes es de mas de dos
                    elif l>2:
                        # Se hace la misma validacion que con los dias porque a veces se ponen dias en el campo de mes
                        # y mes en el campo de dias, de acuerdo al formato que se maneje.
                        list_set = self.set_list(date[i])
                        if len(list_set) == 1:
                            date_dict['date_month'] = str(str(list_set[0])+str(list_set[0]))
                        else:
                            date_dict['date_month'] = str(str(list_set[0])+str(list_set[1]))
                # Si es el año
                if i == 2:
                    # El dataset entero era de 2021, por lo tanto se trabajó en base a este año. Igualmente el código
                    # es facilmente escalable para los próximos años.
                    # Si la longitud de los numeros del año es de 0 o 1, se completa con 2021 automáticamente.
                    if len(date[i]) == 0 or len(date[i]) == 1:
                        date_dict['date_year'] = 2021
                    # Si la longitud de los numeros del año es de 2
                    elif len(date[i]) == 2:
                        # Si se detecta un 0 es muy probable que haya sido escrito el 2021 de manera completa y no haya
                        # detectado los otros dos numeros, entonces se agrega 2021
                        if 0 in date[i]:
                            date_dict['date_year'] = 2021
                        # Si no existe 0, es muy probable que solo se haya escrito 21 como año entonces se agrega el primer
                        # y el segundo numero
                        else:
                            date_dict['date_year'] = str(str(date[i][0])+str(date[i][1]))
                    # Si la longitud de los numeros detectados es de mas de 2
                    elif len(date[i]) > 2:
                        # Se convierte la lista a una lista sin repeticiones
                        list_set = self.set_list(date[i])

                        # Si el segundo numero detectado es un 2, es muy probable que haya sido detectado el 0 primero
                        # o sea el segundo 2 de la lista, entonces se deduce que es 2021
                        if date[i][1] == 2:
                            date_dict['date_year'] = 2021
                        # Si la lista sin numeros repetidos tiene solo dos numeros y no hay un 0, entonces se deduce que
                        # han escrito solo 21
                        elif len(list_set) == 2 and 0 not in list_set:
                            date_dict['date_year'] = 21
                        # Si no, entonces es 2021
                        else:
                            date_dict['date_year'] = 2021

            # Se imprimen los resultados
            print(date_dict)

            # Se actualizan los datos en el dataframe del csv de model_output.csv importado mediante pandas
            model_output.loc[model_output['id']==date_dict['id'], 'date_day'] = date_dict['date_day']
            model_output.loc[model_output['id']==date_dict['id'], 'date_month'] = date_dict['date_month']
            model_output.loc[model_output['id']==date_dict['id'], 'date_year'] = date_dict['date_year']

        # Se exportan las salidas al model_output.csv
        model_output.to_csv('./output/model_output.csv', index=False)
        model_output.to_excel('./output/model_output.xlsx', index=False)
