# -*- coding: utf-8 -*-
"""
Detector de Emociones en Tiempo Real
Autor: TuNombre
Fecha: 2023
Descripción: 
    - Entrena un modelo CNN para reconocimiento de emociones
    - Implementa detección en tiempo real con la cámara
Requisitos:
    - TensorFlow/Keras
    - OpenCV
    - Numpy
"""

import os
import cv2
import numpy as np
from time import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# ====================== CONFIGURACIÓN ======================
DATASET_PATH = "c:/Users/ACER/Documents/deteccion de emociones"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# Parámetros del modelo
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0001
MODEL_NAME = "modelo_emociones.h5"

# Traducción de emociones al español
EMOTION_TRANSLATION = {
    "angry": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "happy": "Feliz",
    "neutral": "Neutral",
    "sad": "Triste",
    "surprise": "Sorpresa"
}

EMOTIONS = list(EMOTION_TRANSLATION.keys())

# ================== PREPARACIÓN DE DATOS ===================
def prepare_data():
    print("\n[INFO] Cargando y preparando datos...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    
    print("[INFO] Clases encontradas:", train_generator.class_indices)
    return train_generator, test_generator

# ================== CONSTRUCCIÓN DEL MODELO ================
def build_model(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

# ===================== ENTRENAMIENTO =======================
def train_model(model, train_generator, test_generator):
    print("\n[INFO] Entrenando modelo...")
    
    start_time = time()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE,
        verbose=1
    )
    
    print(f"\n[INFO] Tiempo de entrenamiento: {time() - start_time:.2f} segundos")
    model.save(MODEL_NAME)
    print(f"[INFO] Modelo guardado como {MODEL_NAME}")
    return history

# ================ EVALUACIÓN DEL MODELO ====================
def evaluate_model(model, test_generator):
    print("\n[INFO] Evaluando modelo...")
    
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    translated_emotions = [EMOTION_TRANSLATION[e] for e in EMOTIONS]
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred_classes, target_names=translated_emotions))

# ============== DETECCIÓN EN TIEMPO REAL ==================
def real_time_detection(model):
    print("\n[INFO] Iniciando detección en tiempo real...")
    print("[INFO] Presiona 'q' para salir")
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, IMG_SIZE)
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            preds = model.predict(roi)[0]
            emotion_key = EMOTIONS[np.argmax(preds)]
            emotion = EMOTION_TRANSLATION[emotion_key]
            prob = np.max(preds)
            
            cv2.putText(frame, f"{emotion}: {prob:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Detector de Emociones', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ==================== EJECUCIÓN PRINCIPAL =================
if __name__ == "__main__":
    # Preparar datos
    train_gen, test_gen = prepare_data()
    
    # Construir modelo
    model = build_model((48, 48, 1), len(EMOTIONS))
    
    # Verificar si el modelo ya está entrenado
    if os.path.exists(MODEL_NAME):
        print("\n[INFO] Cargando modelo pre-entrenado...")
        model = load_model(MODEL_NAME)
        evaluate_model(model, test_gen)  # Evaluación opcional
    else:
        print("\n[INFO] Entrenando nuevo modelo...")
        train_model(model, train_gen, test_gen)
    
    # Iniciar detección en tiempo real
    real_time_detection(model)