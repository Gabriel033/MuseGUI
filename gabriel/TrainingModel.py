import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Lista de archivos CSV proporcionados
file_paths = glob.glob('recordings/*.csv')

# Definir la frecuencia de muestreo (256 Hz)
sampling_rate = 256

# Duración de la ventana en segundos (1 segundo antes y 1 segundo después del estímulo)
window_duration = 2  # 2 segundos
half_window = window_duration // 2  # 1 segundo antes y 1 segundo después
window_size = window_duration * sampling_rate

# Activate data augmentation
data_augmentation_flag = True

segments = []
labels = []

# Función para encontrar el índice del valor más cercano a un timestamp dado
def find_nearest_timestamp(df, value):
    idx = (abs(df['timestamps'] - value)).idxmin()
    return idx

# Función para crear un filtro de paso banda
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

# Crear un escalador
scaler = StandardScaler()

# Función para normalizar los datos
def normalize_data(data):
    return scaler.fit_transform(data)

# Data augmentation function to add Gaussian noise
def add_gaussian_noise(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def time_shift(signal, shift_max=100):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift, axis=0)

def amplitude_scaling(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale

def time_inversion(signal):
    return np.flip(signal, axis=0)

def jitter(signal, sigma=0.01):
    return signal + np.random.normal(0, sigma, signal.shape)

# Iterar sobre cada archivo CSV para extraer y etiquetar segmentos
for file_path in file_paths:
    df = pd.read_csv(file_path)

    # Aplicar el filtrado de paso banda
    filtered_data = bandpass_filter(df.iloc[:, 1:5].values, lowcut=0.5, highcut=50.0, fs=sampling_rate)

    # Normalizar los datos de EEG
    normalized_data = normalize_data(filtered_data)

    df_filtered = pd.DataFrame(normalized_data, columns=df.columns[1:5])

    # Identificar el tipo de estímulo a partir del nombre del archivo o un marcador específico
    if 'EEG_Izquierda' in file_path:
        label = 'IZQUIERDA'
    elif 'EEG_Derecha' in file_path:
        label = 'DERECHA'
    elif "EEG_Reposo" in file_path:
        label = "REPOSO"
    elif "EEG_Arriba" in file_path:
        label = "ARRIBA"
    elif "EEG_Abajo" in file_path:
        label = "ABAJO"
    else:
        continue  # Saltar archivos que no coincidan con ningún estímulo conocido

    # Determinar la duración total de la grabación
    total_duration = round(df['timestamps'].max())

    # Calcular el segundo del estímulo basado en la duración de la grabación
    if total_duration > 12:
        stimulus_time = 9
    else:
        stimulus_time = 4

    start_time = stimulus_time - half_window
    end_time = stimulus_time + half_window
    start_idx = find_nearest_timestamp(df, start_time)
    end_idx = find_nearest_timestamp(df, end_time)

    segment = df_filtered.iloc[start_idx:end_idx]

    if segment.shape[0] < window_size:
        pad_length = window_size - segment.shape[0]
        segment_padded = np.pad(segment.values, ((0, pad_length), (0, 0)), mode='constant')
    elif segment.shape[0] > window_size:
        segment_padded = segment.values[:window_size]
    else:
        segment_padded = segment.values

    segments.append(segment_padded)
    labels.append(label)

    # Apply data augmentation
    if data_augmentation_flag:
        augmented_segments = [
            add_gaussian_noise(segment_padded),
            time_shift(segment_padded),
            amplitude_scaling(segment_padded),
            time_inversion(segment_padded),
            jitter(segment_padded)
        ]

        for aug_segment in augmented_segments:
            segments.append(aug_segment)
            labels.append(label)

label_counts = pd.Series(labels).value_counts()
print(f"Balance de clases:\n{label_counts}")

# Convertir listas a arrays numpy
X = np.array(segments)
y = np.array(labels)

# Verificar la forma de X antes del reshape
print(f"Shape of X before reshaping: {X.shape}")

# Redimensionar los datos para la CNN
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Codificar las etiquetas
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

# Definir la validación cruzada con k-fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Variables para guardar los resultados
cvscores = []

for train, test in kfold.split(X, y):
    # Definir la arquitectura de la CNN simplificada
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 1)))
    model.add(Conv2D(64, (3, 1), activation='relu'))
    model.add(MaxPooling2D((2, 1)))
    model.add(Conv2D(128, (3, 1), activation='relu'))
    model.add(MaxPooling2D((2, 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))  # Agregar dropout para evitar sobreajuste
    model.add(Dense(y.shape[1], activation='softmax'))

    # Compilar el modelo con una tasa de aprendizaje modificada
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    # Entrenar el modelo
    history = model.fit(X[train], y[train], epochs=50, batch_size=32, validation_data=(X[test], y[test]), callbacks=[early_stopping, reduce_lr])

    # Evaluar el modelo
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f"Fold accuracy: {scores[1] * 100}%")
    cvscores.append(scores[1] * 100)

# Después de la validación cruzada, se puede calcular la media y la desviación estándar del accuracy
print(f"Mean accuracy: {np.mean(cvscores)}% (+/- {np.std(cvscores)}%)")