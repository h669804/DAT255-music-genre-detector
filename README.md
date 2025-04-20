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

- tensorflow (for Keras and neural network models)
- numpy (for numerical operations)
- pandas (for CSV feature handling in FCNN)
- matplotlib (for visualization)
- librosa (for audio processing)
- scikit-learn (for train-test splitting, metrics)
- seaborn (for enhanced visualizations)
- scipy (for signal processing, e.g., uniform_filter1d)

## Reproduction Guide

# 1. Clone the repository

Clone the repository into you local storage.
Clone link can be found in the green "Code" button at the top right of the page.

# 2. Download the dataset

**Local**
In order to run the code you are acquired to download the [dataset](<(https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)>)
Make sure to put the dataset into the /data folder in the repository.

Alternatively use Kaggle and adjust the file file paths.

**NOTE!** the audio file: 'jazz.00054.wav' seems to be corrupted. The notebooks should have exception handling for htis, but if you encounter some problems with this, simply delete it.

# 3. Download required packages

**Local**
Download the necessary packages using the follwing command:

```bash
pip install -r requirements.txt
```

**External**
Alternatively use 3rd party tools such as [Kaggle](www.kaggle.com).

# 4. Open and run the notebooks

**Local**
Navigate to your
Order does not matter, since each notebook individually handles the whole pipeline from loading the data, to preprocessing, training and evaluating.
