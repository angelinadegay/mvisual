import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Load and preprocess audio data
def load_data(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    scaler = StandardScaler()
    scaled_mfccs = scaler.fit_transform(mfccs.T)
    return scaled_mfccs, scaler

# Create and train the model
def train_model(data, labels):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(data.shape[1],)),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')  # Example output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data, labels, epochs=10)
    return model

# Example usage
data, scaler = load_data('sample-12s.wav')
labels = np.random.randint(0, 10, size=(data.shape[0],))  # Example labels
model = train_model(data, labels)

# Save the model and scaler
model.save('audio_model.h5')
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
