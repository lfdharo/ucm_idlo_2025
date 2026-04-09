"""
Attention Visualization Module for Speaker Identification

This module provides tools to visualize attention maps from speaker identification
models (wavLM, SpeechBrain, Whisper) overlaid on spectrograms. This is intended to help students understand which frequency regions and time frames the model uses for speaker identification.

Author: Luis F. D'Haro
Date: Apr 7, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
from typing import Optional, Tuple
import torch
from vector_embedding import audio_read


class AttentionVisualizer:
    """Visualize attention maps over spectrograms for better interpretability."""
    
    def __init__(self, model_name: str = 'wavLM'):
        """Initialize attention visualizer.
        
        Args:
            model_name (str): Model name ('wavLM', 'SpeechBrain')
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def display_spectrogram(self,
                           audio_file: str,
                           title: str = "Spectrogram",
                           save_to: Optional[str] = None,
                           sr: int = 16000) -> None:
        """Display a spectrogram of an audio file.
        
        Args:
            audio_file (str): Path to audio file
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
            
        Example:
            >>> visualizer = AttentionVisualizer()
            >>> visualizer.display_spectrogram('audio.wav', title='Speaker SPK1')
        """
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Compute spectrogram
        D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(D, ref=np.max)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 5))
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency (Mel)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spectrogram saved to {save_to}")
        
        plt.show()
    
    def display_spectrogram_with_attention(self,
                                          audio_file: str,
                                          attention_weights: Optional[np.ndarray] = None,
                                          title: str = "Spectrogram with Attention Map",
                                          save_to: Optional[str] = None,
                                          sr: int = 16000,
                                          alpha: float = 0.5) -> None:
        """Display spectrogram with attention weights overlaid.
        
        Args:
            audio_file (str): Path to audio file
            attention_weights (array, optional): Attention weights to overlay
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
            alpha (float): Transparency of attention overlay (0-1)
            
        Example:
            >>> visualizer.display_spectrogram_with_attention(
            ...     'audio.wav',
            ...     attention_weights=attention_map,
            ...     title='wavLM Attention Pattern'
            ... )
        """
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Compute spectrogram
        D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(D, ref=np.max)
        
        # Create figure with spectrogram
        fig, ax = plt.subplots(figsize=(14, 6))
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', 
                                      ax=ax, cmap='viridis')
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')
        
        # Overlay attention if provided
        if attention_weights is not None:
            attention_weights = attention_weights.squeeze()
            
            # Normalize attention weights to [0, 1]
            if attention_weights.max() > 0:
                attention_normalized = attention_weights / attention_weights.max()
            else:
                attention_normalized = attention_weights
            
            # Create colored overlay
            time_frames = attention_normalized.shape[-1]
            freq_frames = S_db.shape[0]
            
            # Resize attention to match spectrogram
            attention_resized = np.interp(
                np.linspace(0, time_frames, len(S_db[0])),
                np.arange(time_frames),
                attention_normalized.mean(axis=0)
            )
            
            # Add attention as overlay
            times = np.linspace(0, len(y)/sr, len(attention_resized))
            ax.plot(times, np.ones_like(times) * freq_frames * 0.9, 
                   linewidth=3, alpha=0.7, color='red', label='Attention Focus')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Frequency (Mel)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        
        if attention_weights is not None:
            ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spectrogram with attention saved to {save_to}")
        
        plt.show()
    
    def display_frequency_focus(self,
                               audio_file: str,
                               attention_weights: Optional[np.ndarray] = None,
                               title: str = "Frequency Region Focus",
                               save_to: Optional[str] = None,
                               sr: int = 16000) -> None:
        """Display which frequency regions the model focuses on.
        
        Args:
            audio_file (str): Path to audio file
            attention_weights (array, optional): Attention weights
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
            
        Example:
            A visualization showing whether the model focuses on:
            - Low frequencies (voice pitch, fundamental frequency)
            - Mid frequencies (formants, speech characteristics)
            - High frequencies (consonants, fricatives)
        """
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Compute spectrogram
        D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(D, ref=np.max)
        
        # Calculate frequency focus
        if attention_weights is not None:
            attention_by_freq = attention_weights.mean(axis=-1).squeeze()
        else:
            attention_by_freq = S_db.mean(axis=1)
            attention_by_freq = np.abs(attention_by_freq - attention_by_freq.min()) / \
                               (attention_by_freq.max() - attention_by_freq.min())
        
        # Create mel scale for y-axis
        mel_freqs = librosa.mel_frequencies(n_mels=len(attention_by_freq))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Spectrogram
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title("Original Spectrogram", fontsize=12, fontweight='bold')
        fig.colorbar(img, ax=ax1, format='%+2.0f dB', label='Power (dB)')
        
        # Attention by frequency
        colors = ['#2ecc71' if w > np.mean(attention_by_freq) else '#95a5a6' 
                 for w in attention_by_freq]
        ax2.barh(mel_freqs, attention_by_freq, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Attention Weight', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency (Mel)', fontsize=11, fontweight='bold')
        ax2.set_title("Model Focus by Frequency", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Frequency focus plot saved to {save_to}")
        
        plt.show()
    
    def display_temporal_focus(self,
                              audio_file: str,
                              attention_weights: Optional[np.ndarray] = None,
                              title: str = "Temporal Focus (Which parts of the audio?)",
                              save_to: Optional[str] = None,
                              sr: int = 16000) -> None:
        """Display which time frames the model focuses on.
        
        Useful for forensic analysis: identifies which parts of the recording
        are most relevant for speaker identification.
        
        Args:
            audio_file (str): Path to audio file
            attention_weights (array, optional): Attention weights
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
            
        Example:
            >>> visualizer.display_temporal_focus(
            ...     'audio.wav',
            ...     title='Which parts of the recording identify the speaker?'
            ... )
        """
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Compute spectrogram
        D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(D, ref=np.max)
        
        # Calculate temporal focus
        if attention_weights is not None:
            temporal_focus = attention_weights.mean(axis=-2).squeeze()
        else:
            temporal_focus = S_db.mean(axis=0)
            temporal_focus = np.abs(temporal_focus - temporal_focus.min()) / \
                            (temporal_focus.max() - temporal_focus.min())
        
        # Create time axis
        times = librosa.frames_to_time(np.arange(len(temporal_focus)), sr=sr)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})
        
        # Spectrogram
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title("Audio Spectrogram", fontsize=12, fontweight='bold')
        fig.colorbar(img, ax=ax1, format='%+2.0f dB', label='Power (dB)')
        
        # Temporal focus
        colors = ['#e74c3c' if w > np.mean(temporal_focus) else '#3498db' 
                 for w in temporal_focus]
        ax2.bar(times, temporal_focus, width=(times[1]-times[0]), color=colors, 
               edgecolor='black', linewidth=0.5, label='Attention Weight')
        ax2.set_ylabel('Attention', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax2.set_title("Model Focus Over Time", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Temporal focus plot saved to {save_to}")
        
        plt.show()
    
    def compare_two_speakers(self,
                            audio_file_1: str,
                            audio_file_2: str,
                            title: str = "Speaker Comparison",
                            save_to: Optional[str] = None,
                            sr: int = 16000) -> None:
        """Compare spectrograms of two speakers side-by-side.
        
        Useful for forensic analysis: visually compare speaker characteristics.
        
        Args:
            audio_file_1 (str): Path to first audio file
            audio_file_2 (str): Path to second audio file
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
            
        Example:
            >>> visualizer.compare_two_speakers('speaker1.wav', 'speaker2.wav')
        """
        # Load audio
        y1, _ = librosa.load(audio_file_1, sr=sr)
        y2, _ = librosa.load(audio_file_2, sr=sr)
        
        # Compute spectrograms
        D1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=128)
        S1_db = librosa.power_to_db(D1, ref=np.max)
        
        D2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=128)
        S2_db = librosa.power_to_db(D2, ref=np.max)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        img1 = librosa.display.specshow(S1_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title("Speaker 1", fontsize=12, fontweight='bold')
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        img2 = librosa.display.specshow(S2_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title("Speaker 2", fontsize=12, fontweight='bold')
        fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
        
        fig.suptitle(title + "\n(Compare frequency patterns and formant structures)", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_to}")
        
        plt.show()
    
    def display_model_attention_temporal(self,
                                        audio_file: str,
                                        model_name: Optional[str] = None,
                                        title: str = "Model Attention - Temporal Focus (Real Attention Weights)",
                                        save_to: Optional[str] = None,
                                        sr: int = 16000) -> None:
        """Display temporal attention using REAL attention weights from the model.
        
        Uses AttentionWeightExtractor to get actual model attention instead of fallback.
        Supports: wavLM, Whisper, unispeech, xlsr
        
        Args:
            audio_file (str): Path to audio file
            model_name (str, optional): Model to use. Uses self.model_name if not provided
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
            
        Example:
            >>> visualizer = AttentionVisualizer(model_name='wavLM')
            >>> visualizer.display_model_attention_temporal(
            ...     'audio.wav',
            ...     title='WavLM Real Attention'
            ... )
        """
        try:
            from attention_weight_extraction import AttentionWeightExtractor
            
            if model_name is None:
                model_name = self.model_name
            
            self.logger.info(f"Extracting real attention from {model_name}...")
            
            # Extract attention weights
            extractor = AttentionWeightExtractor(model_name=model_name)
            attention_weights, S_db = extractor.extract_attention_weights(audio_file, sr=sr)
            
            if attention_weights is None:
                self.logger.warning("Could not extract attention weights, using fallback")
                self.display_temporal_focus(audio_file, None, title, save_to, sr)
                return
            
            # Create time axis based on actual audio duration and attention frame count
            y, _ = librosa.load(audio_file, sr=sr)
            total_duration = len(y) / sr
            times = np.linspace(0, total_duration, len(attention_weights))
            
            # Create figure with spectrogram and attention
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Spectrogram
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
            ax1.set_title("Audio Spectrogram", fontsize=12, fontweight='bold')
            fig.colorbar(img, ax=ax1, format='%+2.0f dB', label='Power (dB)')
            
            # Model attention with color gradient
            colors = plt.cm.RdYlGn(attention_weights)
            ax2.bar(times, attention_weights, width=(times[1]-times[0]), 
                   color=colors, edgecolor='black', linewidth=0.5)
            ax2.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
            ax2.set_title(f"{model_name} Model Attention Over Time", fontsize=12, fontweight='bold')
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3, axis='y')
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_to:
                plt.savefig(save_to, dpi=300, bbox_inches='tight')
                self.logger.info(f"✓ Model attention plot saved to {save_to}")
            
            plt.show()
            
        except ImportError:
            self.logger.error("attention_weight_extraction module not available")
            self.display_temporal_focus(audio_file, None, title, save_to, sr)
    
    def display_attention_heatmap(self,
                                 audio_file: str,
                                 model_name: Optional[str] = None,
                                 title: str = "Attention Heatmap (Self-Attention Pattern)",
                                 save_to: Optional[str] = None,
                                 sr: int = 16000) -> None:
        """Display 2D attention heatmap showing how the model attends to different time frames.
        
        A heatmap where:
        - X and Y axes: time frames
        - Color intensity: attention weight between time frames
        - Diagonal: typically strongest (self-attention)
        - Off-diagonal: how much the model attends across time
        
        Args:
            audio_file (str): Path to audio file
            model_name (str, optional): Model to use
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
            
        Example:
            >>> visualizer.display_attention_heatmap(
            ...     'audio.wav',
            ...     model_name='wavLM'
            ... )
        """
        try:
            from attention_weight_extraction import AttentionWeightExtractor
            
            if model_name is None:
                model_name = self.model_name
            
            self.logger.info(f"Extracting attention heatmap from {model_name}...")
            
            # Extract attention heatmap
            extractor = AttentionWeightExtractor(model_name=model_name)
            heatmap = extractor.get_attention_heatmap(audio_file, sr=sr)
            
            if heatmap is None:
                self.logger.warning("Could not extract attention heatmap")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot heatmap
            im = ax.imshow(heatmap, cmap='hot', aspect='auto', interpolation='nearest')
            fig.colorbar(im, ax=ax, label='Attention Weight')
            
            ax.set_xlabel('Time Frame (Attended To)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Time Frame (Attending From)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if save_to:
                plt.savefig(save_to, dpi=300, bbox_inches='tight')
                self.logger.info(f"✓ Attention heatmap saved to {save_to}")
            
            plt.show()
            
        except ImportError:
            self.logger.error("attention_weight_extraction module not available")


# ======================== CONVENIENCE FUNCTIONS ========================

def visualize_spectrogram(audio_file: str, **kwargs) -> None:
    """Quick function to visualize a spectrogram.
    
    Example:
        >>> visualize_spectrogram('audio.wav', title='My Speaker')
    """
    visualizer = AttentionVisualizer()
    visualizer.display_spectrogram(audio_file, **kwargs)


def visualize_speaker_comparison(audio_file_1: str, audio_file_2: str, **kwargs) -> None:
    """Quick function to compare two speakers.
    
    Example:
        >>> visualize_speaker_comparison('speaker1.wav', 'speaker2.wav')
    """
    visualizer = AttentionVisualizer()
    visualizer.compare_two_speakers(audio_file_1, audio_file_2, **kwargs)


def visualize_temporal_focus(audio_file: str, **kwargs) -> None:
    """Quick function to see which parts of audio matter.
    
    Example:
        >>> visualize_temporal_focus('audio.wav')
    """
    visualizer = AttentionVisualizer()
    visualizer.display_temporal_focus(audio_file, **kwargs)
