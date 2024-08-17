import pyaudio
import numpy as np

# Configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048

def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    print(f"Audio Data: {audio_data[:10]}")  # Print first 10 samples for inspection
    return (in_data, pyaudio.paContinue)

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=BUFFER_SIZE,
                    stream_callback=callback)
    stream.start_stream()

    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
