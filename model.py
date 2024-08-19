import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Load and preprocess audio data
def load_data(file_path):
    y, sr = librosa.load(file_path)
    
    # Generate 2 seconds of silence (at the same sample rate)
    silence = np.zeros(int(sr * 2))
    
    # Combine the original audio with the silence
    y_with_silence = np.concatenate((y, silence))
    
    # Extract MFCC features from the combined audio
    mfccs = librosa.feature.mfcc(y=y_with_silence, sr=sr, n_mfcc=13)
    
    # Scale the MFCC features
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

# Generate corresponding labels, with some representing silence
num_silence_frames = int(len(data) * 2 / 12)  # Adjust based on the length of silence added
silence_labels = np.full(num_silence_frames, 9)  # Assuming label '9' represents silence

# Generate labels for the original audio
original_labels = np.random.randint(0, 9, size=(data.shape[0] - num_silence_frames,))  # Example labels

# Combine the labels
labels = np.concatenate((original_labels, silence_labels))

# Train the model
model = train_model(data, labels)

# Save the model and scaler
model.save('audio_modell.h5')
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
