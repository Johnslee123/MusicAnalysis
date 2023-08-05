import librosa
import numpy as np
import tensorflow.keras as keras

# Load the trained model
model = keras.models.load_model('/Users/johnslee/PycharmProjects/MusicAnalysis/saved_model.h5')


# Function to extract MFCC features from audio file
def extract_mfcc(y, sr, num_mfcc=13, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs


# Load and preprocess the audio data
audio, sr = librosa.load('NewJeans_Hurt.wav')
mfccs = extract_mfcc(y=audio, sr=sr, num_mfcc=13, n_fft=2048, hop_length=512)

# Transpose the MFCCs array to match the shape expected by the model
mfccs = mfccs.T  # Transpose the array

# Pad or truncate the time steps to match the expected input shape (130 time steps)
expected_time_steps = 130
if mfccs.shape[0] < expected_time_steps:
    mfccs = np.pad(mfccs, ((0, expected_time_steps - mfccs.shape[0]), (0, 0)), mode='constant')
else:
    mfccs = mfccs[:expected_time_steps, :]

# Reshape the MFCCs to have the shape (1, time_steps, num_mfcc, 1)
mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

# Make a prediction
predicted_probs = model.predict(mfccs)
predicted_class = np.argmax(predicted_probs)

class_names = ['happy songs', 'sad songs']

print(f"Predicted class: {class_names[predicted_class]}")
print(f"Predicted probabilities: {predicted_probs[0]}")
