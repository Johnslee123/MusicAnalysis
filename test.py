import os
import librosa
import math
import json
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import spotipy
import requests
from spotipy.oauth2 import SpotifyClientCredentials

DATASET_PATH = "All KPop"
JSON_PATH_MERGED = 'test.json'
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


# testing for push
song_artist_names = []  # List to store song names and artist names



def download_track_preview(preview_url, track_name, artist_name, save_folder):
    response = requests.get(preview_url)
    if response.status_code == 200:
        file_name = f"{artist_name}_{track_name}.mp3".replace(" ", "_").replace("/", "_")
        file_path = os.path.join(save_folder, file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)
        song_artist_names.append((track_name, artist_name))
        print(f"Downloaded {track_name} by {artist_name} to {file_path}")
        return file_path
    else:
        print(f"Failed to download {track_name} by {artist_name}")
        return None


def save_mfcc(dataset_path, json_path, num_segments=10, n_mfcc=13, n_fft=2048, hop_length=512):
    data = {
        "tracks": []
    }

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    playlist_tracks = sp.playlist_tracks(dataset_path)

    print("\nProcessing playlist")

    for item in playlist_tracks['items']:
        track = item['track']
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        print("Processing:", track_name, "by", artist_name)

        if track['preview_url']:
            audio_file = download_track_preview(track['preview_url'], track_name, artist_name, DATASET_PATH)
            if audio_file is not None:
                signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
                os.remove(audio_file)  # Delete the temporary audio file

                # Get audio features using the track ID
                audio_features = sp.audio_features(track_id)
                if audio_features:
                    audio_features = audio_features[0]  # Take the first element as it contains the data

                # Calculate MFCC features for the entire song
                mfcc = librosa.feature.mfcc(y=signal,
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)

                mfcc = mfcc.T

                track_info = {
                    "track_id": track_id,
                    "mfcc": mfcc.tolist(),
                    "danceability": audio_features.get("danceability"),
                    "energy": audio_features.get("energy"),
                    "loudness": audio_features.get("loudness"),
                    "tempo": audio_features.get("tempo"),
                    "valence": audio_features.get("valence")
                }
                data["tracks"].append(track_info)
                print("Processed", track_name)
            else:
                print(f"Failed to download {track_name} by {artist_name}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


DATA_PATH = "test.json"

def load_data(data_path, max_sequence_length):
    """Loads training dataset from json file.

    :param data_path (str): Path to json file containing data
    :param max_sequence_length (int): Maximum sequence length for MFCC features
    :return X (ndarray): Inputs
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    tracks = data["tracks"]

    X_mfcc = [track["mfcc"] for track in tracks]
    X_mfcc_padded = keras.preprocessing.sequence.pad_sequences(X_mfcc, maxlen=max_sequence_length, padding='post',
                                                               dtype='float32')

    X_other_features = np.array([
        [track["danceability"], track["energy"], track["loudness"], track["tempo"], track["valence"]]
        for track in tracks
    ])

    X_other_features_reshaped = np.repeat(X_other_features[:, np.newaxis, :], max_sequence_length, axis=1)

    X = np.concatenate((X_mfcc_padded, X_other_features_reshaped), axis=2)

    return X


def prepare_datasets(max_sequence_length):
    X = load_data(DATA_PATH, max_sequence_length)

    # Flatten the 3D MFCC sequences to 2D
    X_flattened = X.reshape(X.shape[0], -1)

    # Standardize the features
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_flattened)

    return X_test


def predict(model, X):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    """

    # Perform prediction
    prediction = model.predict(X.reshape(1, -1))

    # Get the predicted label
    predicted_label = np.argmax(prediction, axis=1)

    # Map the predicted label to emotions
    emotion = "happy" if predicted_label == 0 else "sad"

    # Print the predicted probabilities and emotion
    print("Predicted probabilities:", prediction)
    print("Predicted emotion:", emotion)


if __name__ == "__main__":
    max_sequence_length = 100
    client_id = '9fc9ef50f56f42499f85cb66b6c8c3a0'
    client_secret = 'df8d7b7026c84e6db2afab4422f9259d'

    playlist_link = "https://open.spotify.com/playlist/1yzX8r6rEYE6f1s2vdD20V?si=c45bf6042619477b"

    save_mfcc(playlist_link, JSON_PATH_MERGED)
    X_playlist = load_data(JSON_PATH_MERGED, max_sequence_length)

    # Load your trained model (replace 'model.h5' with the actual path to your model)
    model = keras.models.load_model('model.h5')

    X_test = prepare_datasets(max_sequence_length)

    

    # Save song data and load the data for the playlist
   

    print("Number of songs in the playlist:", len(X_playlist))

    num_samples_to_predict = len(X_test)  # Predict for all test samples

    for sample_idx, (song_name, artist_name) in enumerate(song_artist_names):
        X_to_predict = X_test[sample_idx]
        print(f"Predicting song {sample_idx + 1}: {song_name} by {artist_name}")
        predict(model, X_to_predict)

