import os
import librosa
import math
import json
import spotipy
import requests
from spotipy.oauth2 import SpotifyClientCredentials

DATASET_PATH = "All KPop"
JSON_PATH_MERGED = 'kpopdata.json'
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


# testing for push

def download_track_preview(preview_url, track_name, artist_name, save_folder):
    response = requests.get(preview_url)
    if response.status_code == 200:
        file_name = f"{artist_name}_{track_name}.mp3".replace(" ", "_").replace("/", "_")
        file_path = os.path.join(save_folder, file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {track_name} by {artist_name} to {file_path}")
        return file_path
    else:
        print(f"Failed to download {track_name} by {artist_name}")
        return None


def save_mfcc(dataset_path, json_path, label, num_segments=10, n_mfcc=13, n_fft=2048, hop_length=512):
    if os.path.exists(json_path):
        with open(json_path, "r") as fp:
            data = json.load(fp)
    else:
        data = {
            "mapping": {},
            "tracks": []
        }

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    playlist_tracks = sp.playlist_tracks(dataset_path)


    playlist_label = label
    data["mapping"] = {0: {"semantic_label": "happy"}, 1: {"semantic_label": "sad"}}

    print(f"\nProcessing {label} playlist")

    for item in playlist_tracks['items']:
        track = item['track']
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']

        if track['preview_url']:
            audio_file = download_track_preview(track['preview_url'], track_name, artist_name, DATASET_PATH)
            if audio_file is not None:
                signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
                os.remove(audio_file)  # Delete the temporary audio file


                # Get audio features using the track ID
                audio_features = sp.audio_features(track_id)
                if audio_features:
                    audio_features = audio_features[0]  # Take the first element as it contains the data

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        track_info = {
                            "track_id": f"{playlist_label}_{track_id}_{s + 1}",
                            "mfcc": mfcc.tolist(),
                            "label": 0 if label == 'happy' else 1,
                            "danceability": audio_features.get("danceability"),
                            "energy": audio_features.get("energy"),
                            "loudness": audio_features.get("loudness"),
                            "tempo": audio_features.get("tempo"),
                            "valence": audio_features.get("valence")
                        }
                        data["tracks"].append(track_info)
                        print("{}, segment: {}".format(audio_file, s + 1))
        else:
            print(f"Failed to download {track_name} by {artist_name}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    client_id = '9fc9ef50f56f42499f85cb66b6c8c3a0'
    client_secret = 'df8d7b7026c84e6db2afab4422f9259d'

    # Get the links for the happy and sad playlists
    happy_playlist_link = "https://open.spotify.com/playlist/309bHEPuUA1vrUMju9wkSB?si=822800ee8c064188"
    sad_playlist_link = "https://open.spotify.com/playlist/41VqifSaZrrSDAB0x7lhH5?si=0f3ec5b7c19b4b33"

    # Save happy song data
    save_mfcc(happy_playlist_link, JSON_PATH_MERGED, 'happy')
    save_mfcc(sad_playlist_link, JSON_PATH_MERGED, 'sad')