# MusicAnalysis - A Machine Learning Project

## Introduction

Welcome to MusicAnalysis, a Machine Learning project undertaken by John Lee and Edward Lee. In this project, we explored various iterations, faced challenges, utilized various technologies, and finally, developed a capable music analysis model.

## Iterations

## Pre-Iteration: Tutorial

Our model was initially based on Valerio Velardo's Deep Learning (for audio) tutorial, which allowed the user to categorize songs based on genre. Using the Spotify API to gain metadata on various tracks, along with training our model with different groups of music, we were able to create a unique model.

### Iteration 1: Data Collection and Preprocessing

In the first iteration, we focused on using two playlists (one happy, one sad) to see if we could make it distinguish between happy and sad songs. One challenge we ran into initially was having a 97% accuracy on the training data with very poor actual test results. To make a more precise model, we increased the parameters for songs from just MFCC to include energy, tempo, etc.

## Challenges Faced

1. **Data Collection:** Due to copyright restrictions and file conversion issues, there was an initial concern as to where we'd get data, but further inspection into Spotify's API gave us 30 second clips for as many songs as we needed.

## Technologies Used

- Python: The primary programming language for the project.
- TensorFlow: Deep learning frameworks used for model development and training.
- Librosa: A Python library for audio and music analysis, used for feature extraction.

## Final Product Capabilities

The final product of MusicAnalysis is a user-friendly web application capable of real-time music analysis. Its key features include:

1. **Genre Classification:** TBD