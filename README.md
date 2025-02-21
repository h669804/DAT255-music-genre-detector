# DAT255-music-genre-detector

# Music Genre Classification

Authors: Erlend Vitsø, Gard Molegoda, Markus Nedreberg Gjerde

## Dataset

The dataset used for this project consists of:

- **Raw audio files**: Audio clips in time-series format.
- **Spectrograms**: Visual representations of the frequency spectrum over time, generated from the audio files.
- **CSV files**: Features extracted from the audio, split into:
  - **30-second segments**
  - **3-second segments**

You can download the dataset from [GTZAN Music Genre Classification on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download).

## Project Description

The goal of this project is to develop and evaluate three models for classifying music genres:

1. **CNN for Audio**: A Convolutional Neural Network (CNN) that operates on raw audio files (time-series data).
2. **CV for Spectrograms**: A Computer Vision (CV) model that operates on spectrogram images derived from the audio files.
3. **FCNN for CSV Features**: A Fully Connected Neural Network (FCNN) that processes CSV feature data from both 30-second and 3-second audio segments.

Each model will be trained and evaluated separately to determine which approach is most effective for genre classification.

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib (for visualization)
- librosa (for audio processing)
