import gradio as gr # type: ignore
import traceback
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize # type: ignore
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Define genres
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Build the FCNN model architecture
def build_fcnn(input_dim=57, num_classes=10):
    inputs = keras.Input(shape=(input_dim,))
    attention_input = keras.layers.Reshape((input_dim, 1))(inputs)
    attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(attention_input, attention_input)
    attention = keras.layers.Reshape((input_dim,))(attention[:, :, 0])
    x = keras.layers.Concatenate()([inputs, attention])
    x = keras.layers.Dense(512, activation='selu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    residual = x
    x = keras.layers.Dense(512, activation='selu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(512, activation='selu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, residual])
    x = keras.layers.Dense(256, activation='selu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    residual = x
    x = keras.layers.Dense(256, activation='selu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation='selu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, residual])
    x = keras.layers.Dense(128, activation='selu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Preprocessing functions
def preprocess_1d_cnn(audio_data, sample_rate=22050):
    try:
        if audio_data is None or len(audio_data) == 0:
            print("Audio data is empty or None")
            return None
        
        chunk_duration = 4
        overlap = 2
        chunk_samples = int(chunk_duration * sample_rate)
        step = int((chunk_duration - overlap) * sample_rate)
        
        window_size = 75
        audio_data = uniform_filter1d(audio_data, size=window_size, mode='nearest')
        
        expected_samples = 30 * sample_rate
        if len(audio_data) > expected_samples:
            audio_data = audio_data[:expected_samples]
        elif len(audio_data) < expected_samples:
            audio_data = np.pad(audio_data, (0, expected_samples - len(audio_data)), 'constant')
        
        chunks = []
        num_chunks = (expected_samples - chunk_samples) // step + 1
        
        for i in range(num_chunks):
            start = i * step
            end = start + chunk_samples
            chunk = audio_data[start:end]
            if len(chunk) == chunk_samples:
                chunks.append(chunk)
        
        chunks = np.array(chunks)[..., np.newaxis]
        return chunks
    except Exception as e:
        print(f"Error preprocessing for 1D CNN: {str(e)}")
        traceback.print_exc()
        return None

def preprocess_2d_cnn(audio_data, sample_rate=22050, target_shape=(150, 150), chunk_duration=4, overlap_duration=2):
    try:
        if audio_data is None or len(audio_data) == 0:
            print("Audio data is empty or None")
            return None
        
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        step_size = chunk_samples - overlap_samples
        
        data = []
        num_chunks = max(1, int(np.ceil((len(audio_data) - chunk_samples) / step_size)) + 1)
        
        for i in range(num_chunks):
            start = i * step_size
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram)
        
        result = np.array(data)
        return result
    except Exception as e:
        print(f"Error preprocessing for 2D CNN: {str(e)}")
        traceback.print_exc()
        return None

def preprocess_fcnn(audio_data, sample_rate=22050, scaler=None, use_3sec_chunks=True):
    try:
        if audio_data is None or len(audio_data) == 0:
            print("Audio data is empty or None")
            return None
            
        
        # If using 3-second chunks, split into ten 3-second segments
        if use_3sec_chunks:
            chunk_duration = 3
            chunk_samples = int(chunk_duration * sample_rate)
            num_chunks = 10
            chunks = []
            for i in range(num_chunks):
                start = i * chunk_samples
                end = min(start + chunk_samples, len(audio_data))
                chunk = audio_data[start:end]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
                chunks.append(chunk)
        else:
            chunks = [audio_data]
        
        # Aggregate features across chunks
        all_features = []
        for chunk in chunks:
            features = []
            # Skip length feature
            
            # Features 0-1: chroma_stft_mean, chroma_stft_var
            chroma_stft = librosa.feature.chroma_stft(y=chunk, sr=sample_rate, hop_length=512, n_fft=2048)
            features.append(float(np.mean(chroma_stft)))
            features.append(float(np.var(chroma_stft)))
            
            # Features 2-3: rms_mean, rms_var
            rms = librosa.feature.rms(y=chunk, hop_length=512, frame_length=2048)
            features.append(float(np.mean(rms)))
            features.append(float(np.var(rms)))
            
            # Features 4-5: spectral_centroid_mean, spectral_centroid_var
            spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sample_rate, hop_length=512, n_fft=2048)
            features.append(float(np.mean(spectral_centroid)))
            features.append(float(np.var(spectral_centroid)))
            
            # Features 6-7: spectral_bandwidth_mean, spectral_bandwidth_var
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=sample_rate, hop_length=512, n_fft=2048)
            features.append(float(np.mean(spectral_bandwidth)))
            features.append(float(np.var(spectral_bandwidth)))
            
            # Features 8-9: rolloff_mean, rolloff_var
            rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sample_rate, hop_length=512, n_fft=2048)
            features.append(float(np.mean(rolloff)))
            features.append(float(np.var(rolloff)))
            
            # Features 10-11: zero_crossing_rate_mean, zero_crossing_rate_var
            zero_crossing_rate = librosa.feature.zero_crossing_rate(chunk, hop_length=512, frame_length=2048)
            features.append(float(np.mean(zero_crossing_rate)))
            features.append(float(np.var(zero_crossing_rate)))
            
            # Features 12-15: harmony_mean, harmony_var, perceptr_mean, perceptr_var
            y_harmonic, y_percussive = librosa.effects.hpss(chunk)
            features.append(float(np.mean(y_harmonic)))  # No **2
            features.append(float(np.var(y_harmonic)))
            features.append(float(np.mean(y_percussive)))
            features.append(float(np.var(y_percussive)))
            
            # Feature 16: tempo
            tempo, _ = librosa.beat.beat_track(y=chunk, sr=sample_rate, hop_length=512)
            features.append(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo))
            
            # Features 17-56: mfcc1_mean, mfcc1_var, ..., mfcc20_mean, mfcc20_var
            mfccs = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=20, hop_length=512, n_fft=2048)
            for i in range(20):
                features.append(float(np.mean(mfccs[i])))
                features.append(float(np.var(mfccs[i])))
            
            all_features.append(features)
        
        # Aggregate features
        features = np.mean(all_features, axis=0) if use_3sec_chunks else all_features[0]
        
        feature_count = len(features)
        
        
        features = np.array([float(f) for f in features], dtype=np.float32)
        
        # Apply normalization
        if scaler is None:
            print("WARNING: No scaler provided.")
        else:
            features = scaler.transform(features.reshape(1, -1)).flatten()
        
        result = features.reshape(1, -1)
        return result
    except Exception as e:
        print(f"Error preprocessing for FCNN: {str(e)}")
        traceback.print_exc()
        return None

# Load models and scaler
try:
    print("Loading scaler...")
    scaler = joblib.load('deployment/scaler/feature_scaler.joblib')
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Failed to load scaler: {str(e)}")
    scaler = None
    print("WARNING: No scaler loaded. Normalization may be incorrect.")

try:
    print("Loading 1D CNN model...")
    MODEL_1D = tf.keras.models.load_model('deployment/saved_models/1D_CNN_Adam.h5')
    print("1D CNN model loaded successfully!\n\n")
    
    print("Loading 2D CNN model...")
    MODEL_2D = tf.keras.models.load_model('deployment/saved_models/2D_CNN_Adam.h5')
    print("2D CNN model loaded successfully!\n\n")
    
    print("Loading FCNN model...")
    MODEL_FCNN = build_fcnn(input_dim=57, num_classes=10)
    try:
        MODEL_FCNN.load_weights('deployment/saved_models/FCNN_Adam.h5')
        print("FCNN weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load FCNN weights: {str(e)}")
        traceback.print_exc()
        MODEL_FCNN = None
except Exception as e:
    print(f"Error in model loading: {str(e)}")
    traceback.print_exc()
    MODEL_1D = MODEL_2D = MODEL_FCNN = None

if MODEL_1D is None or MODEL_2D is None or MODEL_FCNN is None:
    print("WARNING: Some models could not be loaded. The system will run with limited functionality.")

description = """
# Music Genre Classification Using Deep Learning
Authors: Erlend Vitsø, Markus Nedreberg Gjerde & Gard Molegoda

This web app demonstrates three deep learning models for music genre classification:

1. **1D CNN Model**: Processes raw audio waveforms
2. **2D CNN Model**: Analyzes mel-spectrograms
3. **FCNN Model**: Uses extracted audio features

## How to Use:
1. Find a song on [Youtube](https://www.youtube.com)
2. Download the song in **.wav** format using a converter, preferably [Yout](https://www.yout.com) which automatically removes silence and posseses normalization features.
3. Upload the audio file
4. Click "Submit" to process
5. View predictions for 10 genres: **rock, classical, jazz, disco, pop, blues, reggae, country, metal, hip-hop**
6. Check preprocessing times for each model (in seconds)

**Note**: Audio is processed in 30-second chunks (as many as possible), with predictions averaged across chunks. Processing time depends on song length (approx. 5 seconds per 30 seconds). Use clear music tracks.

**Disclaimer**: Predictions may vary due to subjective genres.
**Disclaimer**: We do not promote downloading copyrighted music for personal use. Use of this application is for educational purposes only.
"""

def process_audio(audio_file=None):
    try:
        if audio_file is None or audio_file == "":
            return ["No audio file provided", "No audio file provided", "No audio file provided", "N/A", "N/A", "N/A"]
        
        print(f"Using uploaded audio file: {audio_file}")
        audio_path = audio_file
        
        # Load full audio
        print("Loading full audio file...")
        start_time = time.time()
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        audio_load_time = time.time() - start_time
        if y is None or len(y) == 0:
            print("Audio data is empty or None")
            return ["Error loading audio", "Error loading audio", "Error loading audio", "N/A", "N/A", "N/A"]
        
        print(f"Full audio loaded, length: {len(y)} samples ({len(y)/sr:.2f} seconds)")
        
        # Divide into 30-second chunks
        chunk_duration = 30
        chunk_samples = int(chunk_duration * sr)
        num_chunks = max(1, len(y) // chunk_samples)  # As many full 30-second chunks as possible
        audio_chunks = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, len(y))
            chunk = y[start:end]
            if len(chunk) == chunk_samples:
                audio_chunks.append(chunk)
            elif len(chunk) >= chunk_samples // 2:  # Include partial chunk if at least 15 seconds
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
                audio_chunks.append(chunk)
        
        print(f"Created {len(audio_chunks)} 30-second chunks for processing")
        
        results = ["Processing...", "Processing...", "Processing..."]
        processing_times = ["N/A", "N/A", "N/A"]
        
        # 1D CNN
        try:
            if MODEL_1D is not None:
                start_time = time.time()
                all_pred_1d = []
                print("Processing 1D CNN")
                for i, chunk in enumerate(audio_chunks):
                    
                    input_1d = preprocess_1d_cnn(chunk, sample_rate=sr)
                    if input_1d is not None and len(input_1d) > 0:
                        pred_1d = MODEL_1D.predict(input_1d, verbose=0)
                        all_pred_1d.append(pred_1d)
                if all_pred_1d:
                    all_pred_1d = np.concatenate(all_pred_1d, axis=0)
                    avg_pred_1d = np.mean(all_pred_1d, axis=0)
                    
                    # Print detailed prediction information
                    print(f"1D CNN average prediction probabilities:")
                    for i, genre in enumerate(GENRES):
                        print(f"  {genre}: {avg_pred_1d[i]:.4f}")
                    
                    genre_1d = GENRES[np.argmax(avg_pred_1d)]
                    conf_1d = np.max(avg_pred_1d)
                    results[0] = f"{genre_1d} ({conf_1d:.2f})"
                    processing_times[0] = f"{time.time() - start_time:.2f} seconds"
                else:
                    results[0] = "Error preprocessing audio"
                    processing_times[0] = "Error"
            else:
                results[0] = "Model not available"
                processing_times[0] = "N/A"
        except Exception as e:
            print(f"Error with 1D CNN: {str(e)}")
            traceback.print_exc()
            results[0] = "Error in prediction"
            processing_times[0] = "Error"
        
        # 2D CNN
        try:
            if MODEL_2D is not None:
                start_time = time.time()
                all_pred_2d = []
                print("Processing 2D CNN")
                for i, chunk in enumerate(audio_chunks):
                    input_2d = preprocess_2d_cnn(chunk, sample_rate=sr)
                    if input_2d is not None and len(input_2d) > 0:
                        pred_2d = MODEL_2D.predict(input_2d, verbose=0)
                        all_pred_2d.append(pred_2d)
                if all_pred_2d:
                    all_pred_2d = np.concatenate(all_pred_2d, axis=0)
                    avg_pred_2d = np.mean(all_pred_2d, axis=0)
                    
                    # Print detailed prediction information
                    print(f"2D CNN average prediction probabilities:")
                    for i, genre in enumerate(GENRES):
                        print(f"  {genre}: {avg_pred_2d[i]:.4f}")
                    
                    genre_2d = GENRES[np.argmax(avg_pred_2d)]
                    conf_2d = np.max(avg_pred_2d)
                    results[1] = f"{genre_2d} ({conf_2d:.2f})"
                    processing_times[1] = f"{time.time() - start_time:.2f} seconds"
                else:
                    results[1] = "Error preprocessing audio"
                    processing_times[1] = "Error"
            else:
                results[1] = "Model not available"
                processing_times[1] = "N/A"
        except Exception as e:
            print(f"Error with 2D CNN: {str(e)}")
            traceback.print_exc()
            results[1] = "Error in prediction"
            processing_times[1] = "Error"
        
        # FCNN
        try:
            if MODEL_FCNN is not None:
                start_time = time.time()
                all_pred_fcnn = []
                print("Processing FCNN")
                for i, chunk in enumerate(audio_chunks):
                    input_fcnn = preprocess_fcnn(chunk, sample_rate=sr, scaler=scaler, use_3sec_chunks=False)  # Set to True if use_combined=True
                    if input_fcnn is not None:
                        pred_fcnn = MODEL_FCNN.predict(input_fcnn, verbose=0)
                        all_pred_fcnn.append(pred_fcnn)
                if all_pred_fcnn:
                    all_pred_fcnn = np.concatenate(all_pred_fcnn, axis=0)
                    avg_pred_fcnn = np.mean(all_pred_fcnn, axis=0)
                    
                    # Print detailed prediction information
                    print(f"FCNN average prediction probabilities:")
                    for i, genre in enumerate(GENRES):
                        print(f"  {genre}: {avg_pred_fcnn[i]:.4f}")
                    
                    genre_fcnn = GENRES[np.argmax(avg_pred_fcnn)]
                    conf_fcnn = np.max(avg_pred_fcnn)
                    results[2] = f"{genre_fcnn} ({conf_fcnn:.2f})"
                    processing_times[2] = f"{time.time() - start_time:.2f} seconds"
                else:
                    results[2] = "Error preprocessing audio"
                    processing_times[2] = "Error"
            else:
                results[2] = "Model not available"
                processing_times[2] = "N/A"
        except Exception as e:
            print(f"Error with FCNN: {str(e)}")
            traceback.print_exc()
            results[2] = "Error in prediction"
            processing_times[2] = "Error"
        
        return results + processing_times
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        traceback.print_exc()
        return ["Error processing audio", "Error processing audio", "Error processing audio", "N/A", "N/A", "N/A"]
    
# Gradio app
with gr.Blocks() as app:
    gr.Markdown(description)
    with gr.Row():
        input_file = gr.Audio(type="filepath", label="Upload audio file (.wav)")
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
    with gr.Row():
        cnn1d_output = gr.Label(label="1D CNN Prediction")
        cnn2d_output = gr.Label(label="2D CNN Prediction")
        fcnn_output = gr.Label(label="FCNN Prediction")
    with gr.Row():
        cnn1d_time = gr.Label(label="1D CNN Preprocessing Time")
        cnn2d_time = gr.Label(label="2D CNN Preprocessing Time")
        fcnn_time = gr.Label(label="FCNN Preprocessing Time")
    submit_btn.click(
        fn=process_audio,
        inputs=[input_file],
        outputs=[cnn1d_output, cnn2d_output, fcnn_output, cnn1d_time, cnn2d_time, fcnn_time]
    )

if MODEL_1D is None or MODEL_2D is None or MODEL_FCNN is None:
    print("WARNING: Some models could not be loaded. The system will run with limited functionality.")
else:
    print("\n\nAll models loaded successfully! The app will use all three models for prediction.\n\n")

app.launch()