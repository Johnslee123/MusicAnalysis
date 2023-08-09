import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import os


def download_track_preview(preview_url, track_name, artist_name, save_folder):
    response = requests.get(preview_url)
    if response.status_code == 200:

        file_name = f"{artist_name}_{track_name}.wav".replace(" ", "_").replace("/", "_")
        file_path = os.path.join(save_folder, file_name)

        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {track_name} by {artist_name} to {file_path}")
    else:
        print(f"Failed to download {track_name} by {artist_name}")


def get_track_previews(playlist_id, client_id, client_secret):
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

    playlist_tracks = sp.playlist_tracks(playlist_id)

    track_previews = {}
    for item in playlist_tracks['items']:
        track = item['track']
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']

        if track['preview_url']:
            track_previews[track_id] = {
                'track_name': track_name,
                'artist_name': artist_name,
                'preview_url': track['preview_url']
            }
        else:
            track_previews[track_id] = {
                'track_name': track_name,
                'artist_name': artist_name,
                'preview_url': None
            }

    return track_previews


def main():
    client_id = '9fc9ef50f56f42499f85cb66b6c8c3a0'
    client_secret = 'df8d7b7026c84e6db2afab4422f9259d'

    playlist_id = 'https://open.spotify.com/playlist/309bHEPuUA1vrUMju9wkSB?si=822800ee8c064188'

    track_previews = get_track_previews(playlist_id, client_id, client_secret)

    save_folder = "All KPop/happy songs"

    for track_info in track_previews.values():
        if track_info['preview_url']:
            print(f"Downloading 30-second preview for {track_info['track_name']} by {track_info['artist_name']}")
            download_track_preview(track_info['preview_url'], track_info['track_name'], track_info['artist_name'],
                                   save_folder)
        else:
            print(f"No 30-second preview available for {track_info['track_name']} by {track_info['artist_name']}")


if __name__ == '__main__':
    main()
