import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from emotion_model import build_emotion_model

# Definir rutas
dataset_dir = '/path/to/your/dataset'
model_save_path = 'emotion_recognition_model.h5'

# Configurar los generadores de datos con aumentación y validación
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalización
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

# Generadores de entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_generator.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Construir el modelo
model = build_emotion_model(input_shape=(
    224, 224, 3), num_classes=train_generator.num_classes)

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)

# Guardar el modelo entrenado
model.save(model_save_path)
print(f"Modelo guardado en {model_save_path}")
