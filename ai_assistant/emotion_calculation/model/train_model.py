import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from emotion_model import build_emotion_model

# Definir rutas
dataset_dir = 'C:/Users/LENOVO/Desktop/Uni/Quinto/Inteligencia Artificial/Proyecto-IA/ai_assistant/emotion_recognition/data'
model_save_path = 'emotion_recognition_model.h5'

# Configurar los generadores de datos con aumentación moderada
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Reducido para acelerar el procesamiento
    zoom_range=0.1,     # Reducido para acelerar el procesamiento
    horizontal_flip=True,
    validation_split=0.2  # Reservar el 20% de los datos para validación
)

# Generador para entrenamiento
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=64,  # Tamaño de batch aumentado para acelerar el entrenamiento
    class_mode='categorical',
    subset='training'
)

# Generador para validación
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# Construir el modelo
model = build_emotion_model(input_shape=(
    224, 224, 3), num_classes=train_generator.num_classes)

# Configurar Early Stopping para detener el entrenamiento si no hay mejoras
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenar el modelo con Early Stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping]
)

# Guardar el modelo entrenado
model.save(model_save_path)
print(f"Modelo guardado en {model_save_path}")
