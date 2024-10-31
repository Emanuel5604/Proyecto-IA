from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


def build_emotion_model(input_shape=(224, 224, 3), num_classes=7):
    # Cargar el modelo base VGG16 preentrenado en ImageNet
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=input_shape)

    # Congelar la mayoría de las capas del modelo base para reducir el tiempo de entrenamiento
    # Descongelamos solo las últimas 4 capas para fine-tuning
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Construir el modelo personalizado
    model = models.Sequential([
        base_model,  # Modelo base VGG16
        layers.Flatten(),  # Aplanar la salida de VGG16
        layers.Dense(128, activation='relu'),  # Capa densa intermedia
        layers.Dropout(0.5),  # Dropout para prevenir sobreajuste
        # Capa de salida para las emociones
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compilar el modelo con una tasa de aprendizaje baja para el fine-tuning
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
