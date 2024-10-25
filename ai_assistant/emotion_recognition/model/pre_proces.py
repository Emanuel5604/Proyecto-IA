import cv2
import numpy as np


def preprocess_image(image, target_size=(224, 224)):
    # Redimensionar la imagen al tamaño requerido por el modelo
    resized_image = cv2.resize(image, target_size)

    # Normalizar la imagen
    normalized_image = resized_image / 255.0

    # Añadir una dimensión adicional para representar el batch (necesario para la predicción)
    processed_image = np.expand_dims(normalized_image, axis=0)

    return processed_image
