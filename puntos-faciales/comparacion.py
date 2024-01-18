import cv2
import dlib
import numpy as np
import json

def obtener_puntos_referencia_json(id):
    with open('datos_imagenes.json') as f:
        datos_json = json.load(f)

    puntos_referencia_json = datos_json[str(id)]["puntos_referencia"]["adicionales"]
    puntos_referencia = np.array([p["coordenadas"] for p in puntos_referencia_json])

    return puntos_referencia

def combinar_puntos_referencia(puntos_referencia_json, puntos_faciales_normales):
    # Combinar los puntos de referencia JSON con los puntos faciales normales
    puntos_referencia_extendidos = np.concatenate((puntos_referencia_json, puntos_faciales_normales), axis=0)

    return puntos_referencia_extendidos

def calcular_puntaje(imagen, puntos_referencia_extendidos, umbral_porcentaje):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detectar la cara en la imagen
    caras = detector(imagen_gris)
    if not caras:
        raise ValueError("No se encontró cara en la imagen.")

    cara = caras[0]

    # Obtener los puntos faciales de la imagen
    puntos_faciales_imagen = predictor(imagen_gris, cara)

    # Convertir los puntos faciales a un array de NumPy
    puntos_referencia_imagen = np.array([[p.x, p.y] for p in puntos_faciales_imagen.parts()])

    # Puedes ajustar estos índices según las necesidades específicas de tu aplicación
    indices_a_excluir = list(range(36, 48))  # Índices de los puntos de los ojos
    puntos_referencia_imagen = np.delete(puntos_referencia_imagen, indices_a_excluir, axis=0)

    # Comparar con los puntos de referencia extendidos
    porcentaje_coincidencia = comparar_puntos_referencia(puntos_referencia_extendidos, puntos_referencia_imagen, umbral_porcentaje)

    return porcentaje_coincidencia

def comparar_puntos_referencia(puntos_referencia_extendidos, puntos_referencia_imagen, umbral_porcentaje):
    # Calcula la distancia entre los puntos de referencia
    distancia = np.linalg.norm(puntos_referencia_imagen - puntos_referencia_extendidos, axis=1)

    # Calcula el porcentaje de coincidencia
    porcentaje_coincidencia = np.mean(distancia < umbral_porcentaje)

    return porcentaje_coincidencia * 100

def mostrar_puntos_en_imagen(imagen, puntos_referencia_extendidos):
    for (x, y) in puntos_referencia_extendidos:
        cv2.circle(imagen, (x, y), 2, (0, 255, 0), -1)

    # Mostrar la imagen
    cv2.imshow('Imagen con Puntos', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uso de la función
id_referencia = 1  # ID de la imagen con puntos de referencia extendidos en tu JSON

# Obtener puntos de referencia de la imagen de referencia en tu JSON
puntos_referencia_json = obtener_puntos_referencia_json(id_referencia)

# Ruta de la imagen a procesar
ruta_imagen = "./static/Imagenes/nAT.jpg"

# Cargar la imagen
imagen_a_procesar = cv2.imread(ruta_imagen)

# Puedes ajustar este umbral según tus necesidades
umbral_porcentaje = 10

# Obtener puntos faciales normales
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
imagen_gris = cv2.cvtColor(imagen_a_procesar, cv2.COLOR_BGR2GRAY)
caras = detector(imagen_gris)
cara = caras[0]
puntos_faciales_normales = np.array([[p.x, p.y] for p in predictor(imagen_gris, cara).parts()])

# Combinar puntos de referencia JSON con puntos faciales normales
puntos_referencia_extendidos = combinar_puntos_referencia(puntos_referencia_json, puntos_faciales_normales)

# Mostrar la imagen con los puntos
mostrar_puntos_en_imagen(imagen_a_procesar.copy(), puntos_referencia_extendidos)

# Calcular el porcentaje de coincidencia
porcentaje_coincidencia = calcular_puntaje(imagen_a_procesar, puntos_referencia_extendidos, umbral_porcentaje)

if porcentaje_coincidencia >= umbral_porcentaje:
    print(f"La imagen tiene tristeza con un porcentaje de coincidencia de {porcentaje_coincidencia}%")
else:
    print("La imagen no tiene tristeza.")
