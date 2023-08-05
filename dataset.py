import os
import librosa
import librosa.feature
import math
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

DATASET_PATH = "All KPop"
JSON_PATH = "kpopdata.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
PLAYLIST_ID = '7ldp9yPrXzrRpcen0xdDZZ'


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
        "danceability": [],
        "energy": [],
        "loudness": [],
        "tempo": [],
        "valence": []
    }

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    playlist_tracks = sp.playlist_tracks(PLAYLIST_ID)

    for i, (dirpath, dirnames, filesnames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            print("\nProcessing{} ".format(semantic_label))

            for f in filesnames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

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
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s + 1))

    for item in playlist_tracks['items']:
        track = item['track']
        track_id = track['id']

        track_info = sp.audio_features(track_id)
        if track_info and 'danceability' in track_info[0]:
            danceability = track_info[0]['danceability']
            print(f"Track: {track_id}, Danceability: {danceability}")
            data['danceability'].append(danceability)

        if track_info and 'energy' in track_info[0]:
            energy = track_info[0]['energy']
            print(f"Track: {track_id}, Energy: {energy}")
            data['energy'].append(energy)

        if track_info and 'tempo' in track_info[0]:
            tempo = track_info[0]['tempo']
            print(f"Track: {track_id}, Tempo: {tempo}")
            data['tempo'].append(tempo)

        if track_info and 'valence' in track_info[0]:
            valence = track_info[0]['valence']
            print(f"Track: {track_id}, Valence: {valence}")
            data['valence'].append(valence)

        if track_info and 'loudness' in track_info[0]:
            loudness = track_info[0]['loudness']
            print(f"Track: {track_id}, Loudness: {loudness}")
            data['loudness'].append(loudness)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
