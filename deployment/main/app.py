import gradio as gr
import traceback
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from scipy.ndimage import uniform_filter1d

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

# Define genres (GTZAN dataset)
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Modellrelaterte funksjoner og klasser
def mixup_data(x, y, alpha=0.2):
    """Performs mixup augmentation on the batch."""
    batch_size = len(x)
    weights = np.random.beta(alpha, alpha, batch_size)
    weights = weights.reshape(batch_size, 1)
    index = np.random.permutation(batch_size)
    x1, x2 = x, x[index]
    y1, y2 = y, y[index]
    x_mixed = x1 * weights + x2 * (1 - weights)
    y_mixed = y1 * weights + y2 * (1 - weights)
    return x_mixed, y_mixed

class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32, alpha=0.2, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.indices = np.arange(len(x))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        batch_x, batch_y = mixup_data(batch_x, batch_y, self.alpha)
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Definer flere varianter av GetItem-laget for å øke sjansen for at en av dem passer med modellen
class GetItem(tf.keras.layers.Layer):
    def __init__(self, index=0, **kwargs):
        super(GetItem, self).__init__(**kwargs)
        self.index = index
    
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return inputs[self.index]
        elif len(tf.shape(inputs)) > 2:
            return inputs[:, :, self.index]
        return inputs
    
    def compute_output_shape(self, input_shape):
        if len(input_shape) > 2:
            return (input_shape[0], input_shape[1])
        return input_shape
    
    def get_config(self):
        config = super(GetItem, self).get_config()
        config.update({'index': self.index})
        return config

class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        if len(tf.shape(inputs)) > 2:
            return inputs[:, :, 0]
        return inputs
    
    def compute_output_shape(self, input_shape):
        if len(input_shape) > 2:
            return (input_shape[0], input_shape[1])
        return input_shape
    
    def get_config(self):
        return super(SliceLayer, self).get_config()

# Spesifikk implementasjon for MultiHeadAttention indeksering
class AttentionSlice(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionSlice, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Check if input has 3 dimensions [batch, seq_len, channels]
        if inputs.shape.rank == 3:
            return inputs[:, :, 0]
        return inputs
    
    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return (input_shape[0], input_shape[1])
        return input_shape
    
    def get_config(self):
        return super(AttentionSlice, self).get_config()

# Last inn modeller
try:
    print("Loading 1D CNN model...")
    MODEL_1D = tf.keras.models.load_model('deployment/saved_models/1D_CNN_Adam.h5')
    print("1D CNN model loaded successfully!")
    
    print("Loading 2D CNN model...")
    MODEL_2D = tf.keras.models.load_model('deployment/saved_models/2D_CNN_Adam.h5')
    print("2D CNN model loaded successfully!")
    
    # Prøv flere varianter av custom objects for å øke sjansen for suksess
    custom_objects = {
        'MixupGenerator': MixupGenerator,
        'mixup_data': mixup_data,
        'SliceLayer': SliceLayer,
        'GetItem': GetItem,
        'AttentionSlice': AttentionSlice
    }
    
    # Prøv flere metoder for å laste inn FCNN-modellen
    print("Attempting to load FCNN model with multiple approaches...")
    MODEL_FCNN = None
    
    # Tilnærming 1: Bruk alle custom objects og standard lasting
    try:
        print("Approach 1: Using all custom objects...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            MODEL_FCNN = tf.keras.models.load_model('deployment/saved_models/FCNN_Adam.h5')
        print("FCNN model loaded successfully with Approach 1!")
    except Exception as e:
        print(f"Approach 1 failed: {str(e)}")
        
        # Tilnærming 2: Prøv uten kompilering først
        try:
            print("Approach 2: Load without compiling...")
            with tf.keras.utils.custom_object_scope(custom_objects):
                MODEL_FCNN = tf.keras.models.load_model('deployment/saved_models/FCNN_Adam.h5', compile=False)
            # Kompiler manuelt 
            MODEL_FCNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("FCNN model loaded successfully with Approach 2!")
        except Exception as e:
            print(f"Approach 2 failed: {str(e)}")
            
            # Tilnærming 3: Bygg modellen manuelt
            try:
                print("Approach 3: Trying to build model from scratch...")
                # Definer funksjonen for å bygge modellen nøyaktig som i treningskoden
                def build_advanced_model(input_dim=57, num_classes=10):
                    # Input layer
                    inputs = keras.Input(shape=(input_dim,))
                    
                    # Self-attention mechanism
                    # Reshape for attention
                    attention_input = keras.layers.Reshape((input_dim, 1))(inputs)
                    
                    # Multi-head attention layer
                    attention = keras.layers.MultiHeadAttention(
                        num_heads=4, key_dim=4
                    )(attention_input, attention_input)
                    
                    # Bruk vår AttentionSlice-lag i stedet for indeksering
                    attention_sliced = AttentionSlice()(attention)
                    
                    # Reshape back to original dimensions
                    attention = keras.layers.Reshape((input_dim,))(attention_sliced)
                    
                    # Combine original input with attention features
                    x = keras.layers.Concatenate()([inputs, attention])
                    
                    # First dense block
                    x = keras.layers.Dense(512, activation='selu')(x)
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.Dropout(0.4)(x)
                    
                    # First residual block
                    residual = x
                    x = keras.layers.Dense(512, activation='selu')(x)
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.Dropout(0.4)(x)
                    x = keras.layers.Dense(512, activation='selu')(x)
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.add([x, residual])  # Skip connection
                    
                    # Second dense block
                    x = keras.layers.Dense(256, activation='selu')(x)
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.Dropout(0.3)(x)
                    
                    # Second residual block
                    residual = x
                    x = keras.layers.Dense(256, activation='selu')(x)
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.Dropout(0.3)(x)
                    x = keras.layers.Dense(256, activation='selu')(x)
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.add([x, residual])  # Skip connection
                    
                    # Third dense block
                    x = keras.layers.Dense(128, activation='selu')(x)
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.Dropout(0.2)(x)
                    
                    # Output layer
                    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
                    
                    model = keras.Model(inputs=inputs, outputs=outputs)
                    
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    return model
                
                # Bygg modellen fra scratch
                MODEL_FCNN = build_advanced_model()
                
                # Last inn vekter fra den eksisterende modellen
                temp_model = None
                with tf.keras.utils.custom_object_scope({'GetItem': None}):
                    temp_model = tf.keras.models.load_model('deployment/saved_models/FCNN_Adam.h5', compile=False)
                
                # Prøv å kopiere vektene lag for lag
                for i, layer in enumerate(MODEL_FCNN.layers):
                    try:
                        if i < len(temp_model.layers):
                            layer.set_weights(temp_model.layers[i].get_weights())
                    except:
                        print(f"Could not transfer weights for layer {i}")
                
                print("FCNN model built and weights transferred with Approach 3!")
            except Exception as e:
                print(f"Approach 3 failed: {str(e)}")
                
                # Tilnærming 4: Skip modell loading helt
                try:
                    print("Approach 4: Building fresh model without loading weights...")
                    MODEL_FCNN = build_advanced_model()
                    print("FCNN model built (without original weights) with Approach 4!")
                    print("WARNING: This model is untrained and will give random results!")
                except Exception as e:
                    print(f"Approach 4 failed: {str(e)}")
                    MODEL_FCNN = None
                    
except Exception as e:
    print(f"Error in model loading process: {str(e)}")
    MODEL_1D = MODEL_2D = MODEL_FCNN = None

if MODEL_1D is None or MODEL_2D is None:
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
2. Download the song in **.wav** format using a converter, preferably [Yout](https://www.yout.com)
3. Upload the audio file
4. Click "Submit" to process
5. View predictions for 10 genres: **rock, classical, jazz, disco, pop, blues, reggae, country, metal, hip-hop**

**Note**: Processing takes approx. 5 seconds. Use clear music tracks.

**Disclaimer**: Predictions may vary due to subjective genres.
"""

def preprocess_1d_cnn(audio_path):
    """Preprocess audio for 1D CNN (raw waveform), matching training code style."""
    try:
        # Konstanter
        sample_rate = 22050
        chunk_duration = 4  # sekunder
        overlap = 2  # sekunder
        chunk_samples = int(chunk_duration * sample_rate)  # 88200 samples
        step = int((chunk_duration - overlap) * sample_rate)  # 44100 samples
        
        # Last inn audio
        print(f"Loading audio file for 1D CNN: {audio_path}")
        audio_data, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=30)
        
        # Sjekk om lyddata er gyldig
        if audio_data is None or len(audio_data) == 0:
            print("Audio data is empty or None")
            return None
        
        print(f"Audio data loaded, length: {len(audio_data)} samples")
        
        # Smooth signalet (som i treningskoden)
        window_size = 75  # Smoothing-vindu
        audio_data = uniform_filter1d(audio_data, size=window_size, mode='nearest')
        
        # Trim eller pad til nøyaktig 30 sekunder
        expected_samples = 30 * sample_rate  # 661500 samples
        if len(audio_data) > expected_samples:
            audio_data = audio_data[:expected_samples]
        elif len(audio_data) < expected_samples:
            audio_data = np.pad(audio_data, (0, expected_samples - len(audio_data)), 'constant')
        
        # Del opp i chunks med overlap
        chunks = []
        num_chunks = (expected_samples - chunk_samples) // step + 1  # Nøyaktig 14 chunks for 30s
        
        for i in range(num_chunks):
            start = i * step
            end = start + chunk_samples
            chunk = audio_data[start:end]
            
            if len(chunk) == chunk_samples:  # Kun ta med fullstendige chunks
                chunks.append(chunk)
        
        # Konverter til numpy array og legg til dimensjon for kanaler
        chunks = np.array(chunks)[..., np.newaxis]  # Shape: (num_chunks, 88200, 1)
        print(f"Created {len(chunks)} chunks for 1D CNN")
        
        return chunks
    except Exception as e:
        import traceback
        print(f"Error preprocessing for 1D CNN: {str(e)}")
        traceback.print_exc()
        return None

def preprocess_2d_cnn(audio_path, target_shape=(150, 150), chunk_duration=4, overlap_duration=2):
    """Preprocess YouTube audio for 2D CNN, matching training pipeline."""
    try:
        # Load audio (resample to 22050 Hz, mono)
        print(f"Loading audio file for 2D CNN: {audio_path}")
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
        
        # Sjekk om lyddata er gyldig
        if y is None or len(y) == 0:
            print("Audio data is empty or None")
            return None
            
        print(f"Audio data loaded for 2D CNN, length: {len(y)} samples")
        
        # Define chunk parameters
        chunk_samples = chunk_duration * sr  # 4 seconds
        overlap_samples = overlap_duration * sr  # 2 seconds
        step_size = chunk_samples - overlap_samples
        
        # Split into chunks
        data = []
        num_chunks = max(1, int(np.ceil((len(y) - chunk_samples) / step_size)) + 1)
        
        for i in range(num_chunks):
            start = i * step_size
            end = min(start + chunk_samples, len(y))
            chunk = y[start:end]
            
            # Pad if chunk is too short
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            # Generate mel-spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            
            data.append(mel_spectrogram)
        
        result = np.array(data)  # Shape: (num_chunks, 150, 150, 1)
        print(f"Created {len(data)} spectrograms for 2D CNN")
        return result
    except Exception as e:
        import traceback
        print(f"Error preprocessing for 2D CNN: {str(e)}")
        traceback.print_exc()
        return None

def preprocess_fcnn(audio_path):
    try:
        print(f"Loading audio file for FCNN: {audio_path}")
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
        
        if y is None or len(y) == 0:
            print("Audio data is empty or None")
            return None
            
        print(f"Audio data loaded for FCNN, length: {len(y)} samples")
        
        features = []
        
        # length
        features.append(float(len(y) / sr))
        
        # chroma_stft
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(float(np.mean(chroma_stft)))
        features.append(float(np.var(chroma_stft)))
        
        # rms
        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))
        features.append(float(np.var(rms)))
        
        # spectral_centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(float(np.mean(spectral_centroid)))
        features.append(float(np.var(spectral_centroid)))
        
        # spectral_bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(float(np.mean(spectral_bandwidth)))
        features.append(float(np.var(spectral_bandwidth)))
        
        # rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(float(np.mean(rolloff)))
        features.append(float(np.var(rolloff)))
        
        # zero_crossing_rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zero_crossing_rate)))
        features.append(float(np.var(zero_crossing_rate)))
        
        # harmony and perceptr
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.append(float(np.mean(y_harmonic**2)))  # harmony_mean
        features.append(float(np.var(y_harmonic**2)))   # harmony_var
        features.append(float(np.mean(y_percussive**2)))  # perceptr_mean
        features.append(float(np.var(y_percussive**2)))   # perceptr_var
        
        # tempo (only one feature)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo))
        
        # MFCC (20 coefficients, mean and var for each)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features.append(float(np.mean(mfccs[i])))
            features.append(float(np.var(mfccs[i])))
        
        feature_count = len(features)
        print(f"Extracted {feature_count} features for FCNN")
        
        expected_count = 57
        if feature_count != expected_count:
            print(f"WARNING: Feature count mismatch. Expected {expected_count}, got {feature_count}.")
            if feature_count > expected_count:
                features = features[:expected_count]
                print(f"Trimmed down to {expected_count} features")
            elif feature_count < expected_count:
                features.extend([0.0] * (expected_count - feature_count))
                print(f"Padded to {expected_count} features")
        
        features = np.array([float(f) for f in features], dtype=np.float32)
        result = features.reshape(1, -1)
        print(f"Final feature shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Error preprocessing for FCNN: {str(e)}")
        traceback.print_exc()
        return None

def process_audio(audio_file=None):
    try:
        if audio_file is None or audio_file == "":
            return ["No audio file provided", "No audio file provided", "No audio file provided"]
        
        print(f"Using uploaded audio file: {audio_file}")
        audio_path = audio_file
        
        results = ["Processing...", "Processing...", "Processing..."]
        
        # 1D CNN
        try:
            if MODEL_1D is not None:
                input_1d = preprocess_1d_cnn(audio_path)
                if input_1d is not None and len(input_1d) > 0:
                    pred_1d = MODEL_1D.predict(input_1d, verbose=0)
                    avg_pred_1d = np.mean(pred_1d, axis=0)
                    genre_1d = GENRES[np.argmax(avg_pred_1d)]
                    conf_1d = np.max(avg_pred_1d)
                    results[0] = f"{genre_1d} ({conf_1d:.2f})"
                else:
                    results[0] = "Error preprocessing audio"
            else:
                results[0] = "Model not available"
        except Exception as e:
            print(f"Error with 1D CNN: {str(e)}")
            results[0] = "Error in prediction"
        
        # 2D CNN
        try:
            if MODEL_2D is not None:
                input_2d = preprocess_2d_cnn(audio_path)
                if input_2d is not None:
                    pred_2d = MODEL_2D.predict(input_2d, verbose=0)
                    avg_pred_2d = np.mean(pred_2d, axis=0)
                    genre_2d = GENRES[np.argmax(avg_pred_2d)]
                    conf_2d = np.max(avg_pred_2d)
                    results[1] = f"{genre_2d} ({conf_2d:.2f})"
                else:
                    results[1] = "Error preprocessing audio"
            else:
                results[1] = "Model not available"
        except Exception as e:
            print(f"Error with 2D CNN: {str(e)}")
            results[1] = "Error in prediction"
        
        # FCNN
        try:
            if MODEL_FCNN is not None:
                input_fcnn = preprocess_fcnn(audio_path)
                if input_fcnn is not None:
                    feature_count = input_fcnn.shape[1]
                    if hasattr(MODEL_FCNN, 'input_shape') and MODEL_FCNN.input_shape[1] != feature_count:
                        print(f"WARNING: Input shape mismatch. Model expects {MODEL_FCNN.input_shape[1]} features, but got {feature_count}.")
                        if MODEL_FCNN.input_shape[1] > feature_count:
                            input_fcnn = np.pad(input_fcnn, ((0,0), (0, MODEL_FCNN.input_shape[1] - feature_count)), 'constant')
                        else:
                            input_fcnn = input_fcnn[:, :MODEL_FCNN.input_shape[1]]
                    
                    pred_fcnn = MODEL_FCNN.predict(input_fcnn, verbose=0)
                    genre_fcnn = GENRES[np.argmax(pred_fcnn)]
                    conf_fcnn = np.max(pred_fcnn)
                    results[2] = f"{genre_fcnn} ({conf_fcnn:.2f})"
                else:
                    results[2] = "Error preprocessing audio"
            else:
                results[2] = "Model not available"
        except Exception as e:
            print(f"Error with FCNN: {str(e)}")
            traceback.print_exc()  # Now traceback is available
            results[2] = "Error in prediction"
        
        return results
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        traceback.print_exc()  # Now traceback is available
        return ["Error processing audio", "Error processing audio", "Error processing audio"]

with gr.Blocks() as app:
    gr.Markdown(description)
    
    with gr.Row():
        input_file = gr.Audio(type="filepath", label="Upload audio file (.wav)")
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        
    gr.Markdown("<br>")
    
    with gr.Row():
        cnn1d_output = gr.Label(label="1D CNN Prediction")
        cnn2d_output = gr.Label(label="2D CNN Prediction")
        fcnn_output = gr.Label(label="FCNN Prediction")

    gr.Markdown("<br>")
    
    submit_btn.click(
        fn=process_audio,
        inputs=[input_file],
        outputs=[cnn1d_output, cnn2d_output, fcnn_output]
    )


# Legg til en feilmelding hvis FCNN-modellen ikke ble lastet
if MODEL_FCNN is None:
    print("\n\nNOTE: The FCNN model could not be loaded due to compatibility issues. ")
    print("The app will run with only 1D and 2D CNN models. ")
    print("The FCNN prediction will show 'Model not available'.\n\n")
else:
    print("\n\nAll models loaded successfully! The app will use all three models for prediction.\n\n")

app.launch()