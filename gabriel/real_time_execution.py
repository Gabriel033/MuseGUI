
import os
import subprocess
import sys
from PyQt5.QtWidgets import QApplication
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import joblib
import numpy as np
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model
import tensorflow as tf
import logging

# Optimización: Configurar TensorFlow para permitir el crecimiento de la memoria
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Definir el tamaño de la ventana basado en los parámetros del entrenamiento
sampling_rate = 256  # Frecuencia de muestreo en Hz
window_duration = 2  # Duración de la ventana en segundos (igual que en el entrenamiento)
window_size = window_duration * sampling_rate  # Tamaño de la ventana en muestras (512)

# Función para crear el filtro de paso banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Función para aplicar el filtro de paso banda
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def normalize_data(data):
    return scaler.transform(data)

# Función para conectar al stream EEG
def connect_to_eeg_stream(timeout=10):
    print("Buscando un stream EEG...")
    streams = resolve_byprop('type', 'EEG', timeout=timeout)
    if not streams:
        raise RuntimeError("No se encontró ningún stream EEG. Asegúrate de que tu dispositivo EEG esté transmitiendo.")
    print("Conectado al stream EEG")
    return StreamInlet(streams[0])

# Función para recolectar datos en tiempo real y hacer predicciones
def real_time_prediction(model, window_size):
    try:
        inlet = connect_to_eeg_stream()

        # Ventana deslizante de datos EEG
        eeg_buffer = []
        confidence_threshold = 0.8  # Umbral para predicciones aceptables
        high_confidence_threshold = 0.9  # Umbral para predicciones confiables
        neutral_zone_lower = 0.5  # Umbral inferior para la zona neutra
        neutral_zone_upper = 0.8  # Umbral superior para la zona neutra

        while True:
            # Obtener datos del inlet
            chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=window_size)
            if timestamps:
                eeg_buffer.extend(chunk)

                # Mantener la ventana deslizante de tamaño fijo
                if len(eeg_buffer) > window_size:
                    eeg_buffer = eeg_buffer[-window_size:]

                # Preprocesar y hacer predicciones si hay suficientes datos
                if len(eeg_buffer) == window_size:
                    eeg_data = np.array(eeg_buffer)

                    # Aplicar el mismo preprocesamiento que usaste para entrenar el modelo
                    filtered_data = bandpass_filter(eeg_data, lowcut=0.5, highcut=50.0, fs=sampling_rate)
                    normalized_data = normalize_data(filtered_data)

                    # Redimensionar los datos para el modelo CNN
                    reshaped_data = normalized_data.reshape(1, normalized_data.shape[0], normalized_data.shape[1], 1)

                    # Hacer la predicción
                    prediction = model.predict(reshaped_data, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)
                    predicted_label = encoder.inverse_transform(predicted_class)
                    max_confidence = np.max(prediction)

                    # Obtener las etiquetas predichas y sus niveles de confianza
                    predicted_classes = encoder.inverse_transform(np.arange(prediction.shape[1]))
                    confidences = prediction[0]

                    # Evaluar la confianza de la predicción y dar la salida final
                    if max_confidence >= high_confidence_threshold or (
                        confidence_threshold <= max_confidence < high_confidence_threshold and not (
                            neutral_zone_lower <= max_confidence < neutral_zone_upper)):
                        # Imprimir las etiquetas y sus niveles de confianza en formato de tabla
                        print(f"{' | '.join(predicted_classes)}")
                        print(f"{' | '.join([f'{confidence:.4f}' for confidence in confidences])}")
                        # Salida final
                        print(f"---> Predicción final: {predicted_label[0]} (Confianza: {max_confidence:.4f})")
                    else:
                        # Continuar si la predicción no es confiable
                        continue

    except Exception as e:
        logging.error(f"Error durante la predicción en tiempo real: {e}")
        print("Reiniciando conexión con el dispositivo EEG...")
        real_time_prediction(model, window_size)

# Iniciar BlueMuse (si es necesario)
subprocess.call('start bluemuse:', shell=True)

# Crear la aplicación PyQt5 (si es necesario)
app = QApplication([sys.argv])

# Cargar tu modelo previamente entrenado
model = load_model('Modelo/modelo_clasificación_eeg162.keras')
encoder = joblib.load('Modelo/label_encoder162.pkl')
scaler = joblib.load('Modelo/scaler162.pkl')

# Iniciar la predicción en tiempo real
real_time_prediction(model, window_size)
