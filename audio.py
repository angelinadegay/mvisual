import numpy as np
import librosa
import pyaudio
import config
from tensorflow.keras.models import load_model
import pickle

class AudioProcessor:
    def __init__(self):
        self.stream = None
        self.model = load_model('audio_model.h5')
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        self.buffer_size = 2048
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.setup_audio_stream()
        self.expected_mfcc_shape = (13, 20)  # Expected shape (n_mfcc, n_frames)

    def setup_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=config.SAMPLE_RATE,
                                  input=True,
                                  stream_callback=self.callback)
        self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status):
        y = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer = np.roll(self.audio_buffer, -len(y))
        self.audio_buffer[-len(y):] = y
        print(f"Audio Buffer: {self.audio_buffer[:10]}")  # Debugging print statement to inspect audio buffer
        return (in_data, pyaudio.paContinue)

    def amplify_signal(self, signal, gain=100.0):  # Increased gain significantly to amplify signal more
        return signal * gain

    def get_audio_features(self):
        if len(self.audio_buffer) < self.buffer_size:
            return None
        y = self.audio_buffer
        y = self.amplify_signal(y)

        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=config.SAMPLE_RATE)
        mfccs = librosa.feature.mfcc(y=y, sr=config.SAMPLE_RATE, n_mfcc=self.expected_mfcc_shape[0])

        # Pad or truncate MFCCs to ensure consistent shape
        if mfccs.shape[1] < self.expected_mfcc_shape[1]:
            mfccs = np.pad(mfccs, ((0, 0), (0, self.expected_mfcc_shape[1] - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :self.expected_mfcc_shape[1]]

        print(f"RMS: {rms}, Chroma STFT: {chroma_stft}, MFCCs: {mfccs}")  # Debugging print statement
        return {'rms': rms, 'chroma_stft': chroma_stft, 'mfccs': mfccs}

    def predict(self):
        features = self.get_audio_features()
        if features:
            mfccs = features['mfccs']
            print(f"MFCCs Shape: {mfccs.shape}")  # Debugging print statement
            if mfccs.shape != self.expected_mfcc_shape:
                return None
            scaled_mfccs = self.scaler.transform(mfccs.T)
            predictions = self.model.predict(scaled_mfccs)
            print("Predictions:", predictions)  # Debugging print statement
            return predictions
        return None
