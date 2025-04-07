import numpy as np
import librosa
import torch
import logging
from typing import Optional, Union
from python_speech_features import fbank
from random import choice

# Constants
SAMPLE_RATE = 16000  # The sampling frequency of the audio files
NUM_FRAMES = 300  # Number of frames used to create a vector embedding
NUM_FBANKS = 64  # Number of filters along the spectogram

def sample_from_mfcc(mfcc: np.ndarray, max_length: int) -> np.ndarray:
    """Returns a subset of max_length of the MFCC data.
    
    Args:
        mfcc (np.ndarray): MFCC features
        max_length (int): Maximum length of the output
        
    Returns:
        np.ndarray: Sampled MFCC features
    """
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r: r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)

    sample = np.expand_dims(s, axis=-1)
    logging.debug(f'sample_from_mfcc: {sample}')
    return sample

def pad_mfcc(mfcc: np.ndarray, max_length: int) -> np.ndarray:
    """Pad MFCC features to max_length.
    
    Args:
        mfcc (np.ndarray): MFCC features
        max_length (int): Target length
        
    Returns:
        np.ndarray: Padded MFCC features
    """
    if len(mfcc) < max_length:
        mfcc = np.vstack(
            (mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1)))
        )
    logging.debug(f'pad_mfcc: {mfcc}')
    return mfcc

def audio_read(filename: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file.
    
    Args:
        filename (str): Path to audio file
        sample_rate (int): Target sampling rate
        
    Returns:
        np.ndarray: Audio signal
    """
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
    logging.debug(f'audio_read: sr: {sr} | sample_rate: {sample_rate}')
    assert sr == sample_rate
    return audio

def read_mfcc(input_filename: str, sample_rate: int) -> np.ndarray:
    """Read audio file and extract MFCC features.
    
    Args:
        input_filename (str): Path to audio file
        sample_rate (int): Target sampling rate
        
    Returns:
        np.ndarray: MFCC features
    """
    audio = audio_read(input_filename, sample_rate)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    audio_voice_only = audio[offsets[0]: offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    logging.debug(f'read_mfcc: {mfcc}')
    return mfcc

def mfcc_fbank(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract MFCC features and normalize them.
    
    Args:
        signal (np.ndarray): Audio signal
        sample_rate (int): Sampling rate
        
    Returns:
        np.ndarray: Normalized MFCC features
    """
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    frames_features_array = np.array(frames_features, dtype=np.float32)
    logging.debug(f'mfcc_fbank: {frames_features_array}')
    return frames_features_array

def normalize_frames(m: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Apply z-norm normalization.
    
    Args:
        m (np.ndarray): Features to normalize
        epsilon (float): Small constant to avoid division by zero
        
    Returns:
        np.ndarray: Normalized features
    """
    normalized_frames = [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]
    logging.debug(f'normalize_frames: {normalized_frames}')
    return normalized_frames

def exctract_vector_embedding(audio_input: str, model_name: str, model: Optional[object] = None, 
                           feature_extractor: Optional[object] = None) -> np.ndarray:
    """Extract speaker embedding vector from audio file.
    
    Args:
        audio_input (str): Path to audio file
        model_name (str): Name of the model
        model (object, optional): Model to use for embedding extraction
        feature_extractor (object, optional): Feature extractor for specific models
        
    Returns:
        np.ndarray: Speaker embedding vector
    """
    if model_name == 'deepspeaker':
        mfcc = sample_from_mfcc(read_mfcc(audio_input, SAMPLE_RATE), NUM_FRAMES)
        vector_prediction = model.rescnn.predict(np.expand_dims(mfcc, axis=0))

    elif model_name == 'wavLM':
        signal, fs = librosa.load(audio_input, sr=44100)
        signal = librosa.resample(signal, orig_sr=fs, target_sr=16000)
        inputs = feature_extractor(signal, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            vector_prediction = model(**inputs).embeddings
        vector_prediction = torch.nn.functional.normalize(vector_prediction, dim=-1).cpu()

    elif model_name == 'SpeechBrain':
        signal, fs = librosa.load(audio_input, sr=44100)
        signal = torch.from_numpy(librosa.resample(signal, orig_sr=fs, target_sr=16000))
        vector_prediction = model.encode_batch(signal)[0].numpy()

    elif model_name == 'whisper':
        MAX_INPUT_LENGTH = 16000 * 30
        signal, fs = librosa.load(audio_input, sr=44100)
        sample = librosa.resample(signal, orig_sr=fs, target_sr=16000)
        sample_batch = [sample[i:i + MAX_INPUT_LENGTH] for i in range(0, len(sample), MAX_INPUT_LENGTH)]
        vector_prediction = model(sample_batch, sampling_rate=16000, return_tensors="pt").input_features[0].numpy()
        vector_prediction = np.reshape(np.asarray(np.mean(vector_prediction, axis=1)), (1, -1))

    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return vector_prediction 