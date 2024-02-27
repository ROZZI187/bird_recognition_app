import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import librosa
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt

audio_data_path = 'C:\\...\\training_data'
inference_categories = os.listdir(audio_data_path)
category_count = len(inference_categories)

def load_and_preprocess_data(data_dir, classes, target_shape=(200, 200)):
    data = []
    labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram)
                labels.append(i)

    return np.array(data), np.array(labels)

data, labels = load_and_preprocess_data(audio_data_path, inference_categories)
labels = to_categorical(labels, num_classes=len(inference_categories))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

class SaveBestModel(Callback):
    def __init__(self, model_dir='models'):
        super(SaveBestModel, self).__init__()
        self.best_accuracy = 0.63
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy > self.best_accuracy:
            print(f'\nValidation accuracy improved from {self.best_accuracy} to {val_accuracy}. Saving the model.')
            self.best_accuracy = val_accuracy

            # Konwersja modelu do formatu TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()

            model_filename = f'model_val_accuracy_{val_accuracy:.4f}.tflite'
            model_path = os.path.join(self.model_dir, model_filename)
            
            # Zapisz model w formacie TensorFlow Lite
            with open(model_path, 'wb') as f:
                f.write(tflite_model)

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= 0.65:
            print(f"\nValidation accuracy reached the threshold. Stopping training.")
            self.model.stop_training = True

callbacks = [
    MyCallback(),
    SaveBestModel(model_dir='models')
]

def create_optimized_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),  # Zmniejszenie liczby neuronów
        Dropout(0.3),  # Zwiększenie wartości Dropout
        Dense(category_count, activation='softmax')
    ])

    optimizer = Adam(lr=0.0001)  # Zmniejszenie tempa uczenia
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model = create_optimized_model()
history = model.fit(X_train, y_train, epochs=100, verbose=1, batch_size=25, validation_data=(X_test, y_test), callbacks=callbacks)

# Zapisz model w formacie TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Ewaluacja modelu
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Dokładność na zbiorze testowym: ", test_accuracy[1])

# Wygeneruj i wyświetl wykres
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
