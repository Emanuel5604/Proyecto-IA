from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models


def build_emotion_model(input_shape=(224, 224, 3), num_classes=7):
    # Cargar el modelo base VGG16 preentrenado en ImageNet
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=input_shape)
    base_model.trainable = False  # Congelamos las capas del modelo base

    # Construir el modelo personalizado
    model = models.Sequential([
        base_model,  # Modelo base VGG16
        layers.Flatten(),  # Aplanar la salida de VGG16
        layers.Dense(128, activation='relu'),  # Capa densa intermedia
        layers.Dropout(0.5),  # Dropout para prevenir sobreajuste
        # Capa de salida para las emociones
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model
