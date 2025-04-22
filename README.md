# DAT255-music-genre-detector

![Python](https://img.shields.io/badge/python-3.12.4-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

# Music Genre Classification

**Authors:** Erlend Vitsø, Gard Molegoda, Markus Nedreberg Gjerde

## Table of Contents

- [Dataset](#dataset)
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Reproduction Guide](#reproduction-guide)
- [Notebooks](#notebooks)
- [License](#license)

## Dataset

The dataset used for this project consists of:

- **Raw audio files**: Audio clips in time-series format.
- **Spectrograms**: Visual representations of the frequency spectrum over time, generated from the audio files.
- **CSV files**: Features extracted from the audio, split into:
  - **30-second segments**
  - **3-second segments**

The dataset includes 10 music genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

You can download the dataset from [GTZAN Music Genre Classification on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download).

## Project Description

The goal of this project is to develop and evaluate three models for classifying music genres:

1. **CNN for Audio**: A Convolutional Neural Network (CNN) that operates on raw audio files (time-series data).
2. **CV for Spectrograms**: A Computer Vision (CV) model that operates on spectrogram images derived from the audio files.
3. **FCNN for CSV Features**: A Fully Connected Neural Network (FCNN) that processes CSV feature data from both 30-second and 3-second audio segments.

Each model will be trained and evaluated separately to determine which approach is most effective for genre classification.

## Project Structure

```
DAT255-music-genre-detector/
├── data/                    # Dataset files (raw audio, spectrograms, CSV)
├── models/                  # Trained model files
├── notebooks/               # Jupyter notebooks for model development
├── deployment/              # Deployment code
│   └── main/                # Main application
│       └── app.py           # Gradio application
├── requirements.txt         # Required packages
└── README.md                # Project documentation
```

## Requirements

Requirements to run the app (not the notebooks)

- tensorflow (for Keras and neural network models)
- numpy (for numerical operations)
- pandas (for CSV feature handling in FCNN)
- matplotlib (for visualization)
- librosa (for audio processing)
- scikit-learn (for train-test splitting, metrics)
- seaborn (for enhanced visualizations)
- scipy (for signal processing, e.g., uniform_filter1d)
- gradio (for web application)

## Reproduction Guide

This project was initially intended to be deployed online, but problems with integration and limited upload speed prohibited the deployment. The app is therefore designed to be run on your local machine.

**NOTE:** This application was developed using Python 3.12.4

### 1. Clone the repository

Clone the repository into your local storage.

```bash
git clone https://github.com/h669804/DAT255-music-genre-detector.git
```

### 2. Download the dataset

In order to run the code you are required to download the [dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download).

Make sure to put the dataset into the `/data` folder in the repository.
Alternatively, open the notebooks in Kaggle and adjust the file paths.

**NOTE!** The audio file: 'jazz.00054.wav' seems to be corrupted. The notebooks should have exception handling for this, but if you encounter problems, simply delete it.

### 3. Navigate to the project folder

```bash
cd DAT255-music-genre-detector
```

### 4. Download required packages

Download the necessary packages using the pip package installer by entering the following command:

```bash
pip install -r requirements.txt
```

### 5. Run the gradio app

```bash
python deployment/main/app.py
```

The console should display a local URL (typically http://127.0.0.1:7860). Open this URL in your browser to access the application.

## Notebooks

**IMPORTANT:** The notebooks in this repository were developed and run in Kaggle. They use different file paths than what would be expected in the local repository structure. If you want to run these notebooks:

1. Upload them to Kaggle
2. Adjust the file paths as needed for your Kaggle environment
3. Make sure the dataset is accessible in your Kaggle workspace

**NOTE:** Training the 1D CNN (audio) and 2D CNN (spectrogram) models requires significant computational resources. We utilized Kaggle's GPU functionality, which was crucial for training these models within a reasonable timeframe. Without GPU acceleration, training times may be prohibitively long.

Alternatively, if you wish to run the notebooks locally, you will need to:

1. Modify all data loading and saving paths to match your local repository structure
2. Ensure you have access to a GPU for model training (especially for the CNN models)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note:** After running the app, follow the instructions within the application interface for using the music genre classifier.
