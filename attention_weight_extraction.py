"""
Attention Weight Extraction from Speaker Identification Models

This module extracts attention weights directly from models to understand
which parts of the audio they focus on during speaker identification.

Supported Models:
- WavLM (microsoft/wavlm-base-plus-sv)
- Whisper (openai/whisper-tiny.en)
- SpeechBrain (indirect attention analysis)
- Additional models: UniSpeech-SAT, XLS-R (from HuggingFace)

Author: Luis F. D'Haro
Date: Apr 7, 2026
"""

import numpy as np
import torch
import librosa
import logging
from typing import Optional, Tuple, Dict
from pathlib import Path


class AttentionWeightExtractor:
    """Extract attention weights from speaker identification models."""
    
    def __init__(self, model_name: str = 'wavLM', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize attention weight extractor.
        
        Args:
            model_name (str): Model name ('wavLM', 'Whisper', 'unispeech', 'xlsr')
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load model with attention hooks."""
        if self.model_name == 'wavLM':
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
            
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus-sv"
            )
            self.model = WavLMForXVector.from_pretrained(
                "microsoft/wavlm-base-plus-sv",
                output_attentions=True  # Enable attention output
            ).to(self.device)
            self.model.eval()
            self.logger.info("✓ WavLM model loaded with attention outputs")
            
        elif self.model_name == 'Whisper':
            from transformers import WhisperProcessor, WhisperModel
            
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.model = WhisperModel.from_pretrained(
                "openai/whisper-tiny",
                output_attentions=True  # Enable attention output
            ).to(self.device)
            self.model.eval()
            self.logger.info("✓ Whisper model loaded with attention outputs")
            
        elif self.model_name == 'unispeech':
            # UniSpeech-SAT: Good alternative with explicit attention
            from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForXVector
            
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/unispeech-sat-base-plus-sv"
            )
            self.model = UniSpeechSatForXVector.from_pretrained(
                "microsoft/unispeech-sat-base-plus-sv",
                output_attentions=True
            ).to(self.device)
            self.model.eval()
            self.logger.info("✓ UniSpeech-SAT model loaded with attention outputs")
            
        elif self.model_name == 'xlsr':
            # XLS-R: Multilingual model with attention
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
            
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/xlsr-53-56k"
            )
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "facebook/xlsr-53-56k",
                num_labels=512,  # Embedding dimension
                output_attentions=True
            ).to(self.device)
            self.model.eval()
            self.logger.info("✓ XLS-R model loaded with attention outputs")
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def extract_attention_weights(self, 
                                  audio_file: str,
                                  sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        """Extract attention weights from audio.
        
        Args:
            audio_file (str): Path to audio file
            sr (int): Sample rate
            
        Returns:
            tuple: (attention_weights, mel_spectrogram)
                - attention_weights: Shape [time_steps] - average attention across heads/layers
                - mel_spectrogram: Shape [n_mels, time_steps]
        """
        # Load and process audio
        y, _ = librosa.load(audio_file, sr=sr)
        
        # Compute mel spectrogram for reference
        D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(D, ref=np.max)
        
        # Process audio for model
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(self.device)
        
        # Forward pass with attention extraction
        with torch.no_grad():
            if self.model_name == 'wavLM':
                outputs = self.model(input_values, output_attentions=True)
                attention_weights = outputs.attentions  # Tuple of attention tensors
                
            elif self.model_name == 'Whisper':
                outputs = self.model(input_values, output_attentions=True)
                attention_weights = outputs.decoder_attentions
                
            elif self.model_name in ['unispeech', 'xlsr']:
                outputs = self.model(input_values, output_attentions=True)
                attention_weights = outputs.attentions
            
            else:
                attention_weights = None
        
        # Process attention weights
        if attention_weights is not None:
            # attention_weights is a tuple of tensors, one per layer
            # Each tensor shape: [batch_size, num_heads, seq_len, seq_len]
            
            # Average across all layers
            attention_avg = torch.stack([att.squeeze(0).mean(dim=0) for att in attention_weights])  # [num_layers, seq_len, seq_len]
            
            # Average across layers
            attention_avg = attention_avg.mean(dim=0)  # [seq_len, seq_len]
            
            # Get temporal focus by averaging attention pattern
            # Average across both dimensions to get how much each position attends/is attended to
            temporal_attention = attention_avg.mean(dim=0).cpu().numpy()  # [seq_len]
            
            # Normalize to [0, 1]
            if temporal_attention.max() > 0:
                temporal_attention = temporal_attention / temporal_attention.max()
            
            self.logger.info(f"✓ Extracted attention weights: shape {temporal_attention.shape}")
            
            return temporal_attention, S_db
        
        else:
            self.logger.warning("Could not extract attention weights, using spectrogram fallback")
            return None, S_db
    
    def get_attention_heatmap(self, audio_file: str, sr: int = 16000) -> np.ndarray:
        """Get attention heatmap (2D representation).
        
        Shows which parts of the audio attend to which other parts.
        
        Args:
            audio_file (str): Path to audio file
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Attention heatmap [seq_len, seq_len]
        """
        y, _ = librosa.load(audio_file, sr=sr)
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(self.device)
        
        with torch.no_grad():
            if self.model_name == 'wavLM':
                outputs = self.model(input_values, output_attentions=True)
                attention_weights = outputs.attentions
            elif self.model_name == 'Whisper':
                outputs = self.model(input_values, output_attentions=True)
                attention_weights = outputs.decoder_attentions
            elif self.model_name in ['unispeech', 'xlsr']:
                outputs = self.model(input_values, output_attentions=True)
                attention_weights = outputs.attentions
            else:
                return None
        
        if attention_weights is not None:
            # Average across all layers and heads
            attention_avg = torch.stack([att.squeeze(0).mean(dim=0) for att in attention_weights])
            attention_heatmap = attention_avg.mean(dim=0).cpu().numpy()
            
            return attention_heatmap
        
        return None


def extract_temporal_attention(audio_file: str, 
                               model_name: str = 'wavLM',
                               sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to extract attention weights from audio.
    
    Args:
        audio_file (str): Path to audio file
        model_name (str): Model to use
        sr (int): Sample rate
        
    Returns:
        tuple: (attention_weights, mel_spectrogram)
        
    Example:
        >>> attention, spec = extract_temporal_attention('audio.wav', model_name='wavLM')
        >>> print(f"Attention shape: {attention.shape}")
    """
    extractor = AttentionWeightExtractor(model_name=model_name)
    return extractor.extract_attention_weights(audio_file, sr=sr)


def extract_attention_heatmap(audio_file: str,
                             model_name: str = 'wavLM',
                             sr: int = 16000) -> np.ndarray:
    """Extract 2D attention heatmap.
    
    Example:
        >>> heatmap = extract_attention_heatmap('audio.wav')
        >>> print(f"Heatmap shape: {heatmap.shape}")
    """
    extractor = AttentionWeightExtractor(model_name=model_name)
    return extractor.get_attention_heatmap(audio_file, sr=sr)
