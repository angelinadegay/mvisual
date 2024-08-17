import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Example data (replace with your actual data and labels)
data = np.random.rand(100, 10)  # 100 samples, 10 features each
labels = np.random.randint(10, size=100)  # 100 labels, 10 classes

# Train the model
model.fit(data, labels, epochs=10)

# Save the model
model.save('my_model.h5')
