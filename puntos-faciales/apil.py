import dlib
import cv2
import os
import json

# Cargamos el modelo de detector facial de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Obtener índice del punto de inicio y fin de la boca
inicio_boca, fin_boca = 48, 68

# Función para manejar clics en la imagen
def clic(event, x, y, flags, param):
    global puntos_adicionales

    if modo_seleccion and event == cv2.EVENT_LBUTTONDOWN:
        # Modo selección: agregar punto seleccionado
        for px, py, punto_color in puntos_adicionales:
            distancia = ((x - px)**2 + (y - py)**2)**0.5
            if distancia < radio_seleccion:
                puntos_seleccionados.append((px, py, punto_color))
                cv2.circle(imagen, (px, py), 2, (0, 255, 0), -1)
                cv2.imshow("Reconocimiento Facial", imagen)
                break
        else:
            color_punto = (0, 0, 255)  # Color del punto adicional (rojo en BGR)
            puntos_adicionales.append((x, y, color_punto))
            cv2.circle(imagen, (x, y), 2, color_punto, -1)
            cv2.imshow("Reconocimiento Facial", imagen)

# Ruta de tu propia imagen
ruta_imagen = "./static/Imagenes/tristeza2_Ref.png"

# Obtener el nombre de la imagen sin extensión
nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]

# Archivo para almacenar el índice
archivo_indice = os.path.join(os.getcwd(), "indice.txt")

# Leer el último índice utilizado o establecerlo en 1 si no existe
if os.path.exists(archivo_indice):
    with open(archivo_indice, "r") as f:
        contenido = f.read().strip()
        contador_indice = int(contenido) if contenido else 1
else:
    contador_indice = 1

# Diccionario que contendrá la información de cada imagen
datos_imagenes = {}

# Cargar el JSON existente si existe
archivo_json = "datos_imagenes.json"
if os.path.exists(archivo_json):
    with open(archivo_json, "r") as archivo_existente:
        datos_imagenes = json.load(archivo_existente)

# Cargamos tu imagen
imagen = cv2.imread(ruta_imagen)
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectamos caras en la imagen
caras = detector(gris)

# Lista para almacenar todos los puntos de referencia
puntos_adicionales = []

# Lista para almacenar los puntos seleccionados
puntos_seleccionados = []

# Crear una ventana para la imagen y establecer la función de clic
cv2.namedWindow("Reconocimiento Facial")
cv2.setMouseCallback("Reconocimiento Facial", clic)

# Modo de selección
modo_seleccion = True
radio_seleccion = 5  # Radio para considerar la selección

while True:
    for cara in caras:
        # Predicción de los puntos de referencia faciales para cada cara
        puntos_de_referencia = predictor(gris, cara)

        # Mostramos todos los puntos de referencia predeterminados en la imagen
        for i in range(68):
            x, y = puntos_de_referencia.part(i).x, puntos_de_referencia.part(i).y
            cv2.circle(imagen, (x, y), 2, (0, 255, 0), -1)

        # Mostramos los puntos de referencia adicionales en la imagen
        for i in range(inicio_boca, fin_boca):
            x, y = puntos_de_referencia.part(i).x, puntos_de_referencia.part(i).y
            cv2.circle(imagen, (x, y), 2, (0, 255, 0), -1)
            puntos_adicionales.append((x, y, (0, 255, 0)))

        for px, py, punto_color in puntos_adicionales:
            cv2.circle(imagen, (px, py), 2, punto_color, -1)

    cv2.imshow("Reconocimiento Facial", imagen)
    key = cv2.waitKey(1) & 0xFF

    # Si se presiona 's', ingresar la puntuación y el porcentaje
    if key == ord('s'):
        puntuacion = float(input("Ingrese la puntuación: "))
        porcentaje = float(input("Ingrese el porcentaje: "))
        
        # Almacenar todos los puntos de referencia en el diccionario
        datos_imagenes[str(contador_indice)] = {
            "Nombre_Imagen": nombre_imagen,
            "puntos_referencia": {
                "seleccionados": puntos_seleccionados,
                "adicionales": sorted(
                    [{"coordenadas": [px, py], "color": punto_color} for px, py, punto_color in puntos_adicionales],
                    key=lambda x: x["coordenadas"][0]  # Ordenar por la posición x
                ),
            },
            "puntuacion": puntuacion,
            "porcentaje": porcentaje
        }
        
        print(f"Datos guardados en índice {contador_indice}")
        
        # Incrementar el índice
        contador_indice += 1

        # Escribir los datos en el archivo JSON después de salir del bucle
        with open(archivo_json, "w") as archivo_salida:
            json.dump(datos_imagenes, archivo_salida, indent=2)

        # Actualizar el índice en el archivo de índice
        with open(archivo_indice, "w") as archivo_indice:
            archivo_indice.write(str(contador_indice))

        print("Datos guardados en el archivo JSON.")
        cv2.destroyAllWindows()
        break
    # Si se presiona 'q', salir
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
