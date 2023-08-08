import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import numpy as np
import os
import requests
import tensorflow.keras as keras



def download_track_preview(preview_url, track_name, artist_name):
    response = requests.get(preview_url)
    if response.status_code == 200:
        file_name = f"{artist_name}_{track_name}.mp3".replace(" ", "_").replace("/", "_")
        file_path = os.path.join("audio_previews", file_name)

        # Create the directory if it doesn't exist
        os.makedirs("audio_previews", exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {track_name} by {artist_name} to {file_path}")
        return file_path
    else:
        print(f"Failed to download {track_name} by {artist_name}")
        return None


def print_playlist_info(playlist_link):
    # Replace with your own Spotify API credentials
    client_id = '9fc9ef50f56f42499f85cb66b6c8c3a0'
    client_secret = 'df8d7b7026c84e6db2afab4422f9259d'

    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

    playlist_tracks = sp.playlist_tracks(playlist_link)

    print(f"Printing information for playlist: {playlist_link}")
    print("-" * 80)

    for item in playlist_tracks['items']:
        track = item['track']
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        preview_url = track['preview_url']

        label = 'happy'  # Replace with actual label determination

        if preview_url:
            audio_file = download_track_preview(preview_url, track_name, artist_name)
            if audio_file is not None:
                signal, sr = librosa.load(audio_file, sr=None)
                os.remove(audio_file)  # Delete the temporary audio file

                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Extract MFCC data

                # Load the saved model
                model = keras.models.load_model("model.h5")

                # Get the audio features from Spotify
                danceability = sp.audio_features(track_id)[0].get("danceability")
                energy = sp.audio_features(track_id)[0].get("energy")
                loudness = sp.audio_features(track_id)[0].get("loudness")
                tempo = sp.audio_features(track_id)[0].get("tempo")
                valence = sp.audio_features(track_id)[0].get("valence")

                # Reshape the one-dimensional features to match the number of frames in MFCC
                num_frames = mfcc.shape[0]
                num_mfcc_coefficients = mfcc.shape[1]

                danceability = np.full((num_frames, 1), danceability)
                energy = np.full((num_frames, 1), energy)
                loudness = np.full((num_frames, 1), loudness)
                tempo = np.full((num_frames, 1), tempo)
                valence = np.full((num_frames, 1), valence)

                # Repeat other features to match num_mfcc_coefficients
                other_features_repeated = np.repeat(
                    np.concatenate((danceability, energy, loudness, tempo, valence), axis=1),
                    num_mfcc_coefficients, axis=1
                )

                # Concatenate the arrays along axis=1 (features)
                features = np.concatenate((mfcc, other_features_repeated), axis=1)

                # Load the saved model
                model = keras.models.load_model("model.h5")

                # Make a prediction
                prediction = model.predict(features)

                # Get the index with max value
                predicted_index = np.argmax(prediction, axis=1)

                label = 'happy' if predicted_index == 0 else 'sad'

                print(f"Track Info:")
                print(f" - Track ID: {track_id}")
                print(f" - Track Name: {track_name}")
                print(f" - Artist Name: {artist_name}")
                print(f" - Preview URL: {preview_url}")
                print(f" - Label: {label}")
                print(f" - MFCC: {mfcc.tolist()}")
                print(f" - Danceability: {danceability}")
                print(f" - Energy: {energy}")
                print(f" - Loudness: {loudness}")
                print(f" - Tempo: {tempo}")
                print(f" - Valence: {valence}")
                print("-" * 80)


if __name__ == "__main__":
    playlist_link = "https://open.spotify.com/playlist/1yzX8r6rEYE6f1s2vdD20V?si=d9c7f8e6e92b49a9"
    print_playlist_info(playlist_link)
