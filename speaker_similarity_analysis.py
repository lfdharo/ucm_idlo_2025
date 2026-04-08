"""
Speaker Similarity Analysis and Spectral Comparison

This module compares spectrograms of two audio files to identify similar acoustic regions,
useful for forensic analysis and understanding speaker characteristics.

Features:
- Find closest matching enrollment file for a test file
- Compute spectral similarity using multiple methods
- Highlight similar/dissimilar regions
- Visualize temporal and spectral alignment

Author: Luis F. D'Haro
Date: Apr 7, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
from typing import Tuple, List, Dict, Optional
from scipy.spatial.distance import euclidean
from scipy.signal import correlate2d
import os


class SpecificSpeakerComparison:
    """Compare spectrograms of two speakers."""
    
    def __init__(self, model_name: str = 'wavLM'):
        """Initialize speaker comparison.
        
        Args:
            model_name (str): Model name for context
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def find_closest_enrollment(self, 
                               test_file: str,
                               enrollment_dir: str) -> Tuple[str, float]:
        """Find the closest matching enrollment file for a test file.
        
        Uses speaker ID matching and then finds the enrollment file with
        highest similarity score.
        
        Args:
            test_file (str): Path to test audio file
            enrollment_dir (str): Path to enrollment directory
            
        Returns:
            tuple: (best_enrollment_file, similarity_score)
            
        Example:
            >>> closest_file, score = comparison.find_closest_enrollment(
            ...     './test/SPK1_test.wav',
            ...     './enrollment/'
            ... )
            >>> print(f"Closest match: {closest_file} (score: {score:.2%})")
        """
        from vector_embedding import exctract_vector_embedding
        from models import ModelFactory
        
        model, feature_extractor = ModelFactory.create_model(self.model_name)
        
        # Extract speaker ID from test file
        test_speaker = os.path.basename(test_file).split('_')[0]
        
        # Get test embedding
        test_embedding = exctract_vector_embedding(
            test_file, 
            self.model_name,
            model,
            feature_extractor
        )
        test_embedding = test_embedding.reshape(-1)  # Flatten to 1D
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        best_score = -1
        best_file = None
        
        # Find enrollment files from same speaker
        speaker_dir = os.path.join(enrollment_dir, test_speaker)
        if not os.path.exists(speaker_dir):
            raise FileNotFoundError(f"Speaker directory not found: {speaker_dir}")
        
        # Iterate through enrollment files
        for filename in os.listdir(speaker_dir):
            if filename.endswith('.wav'):
                enrollment_file = os.path.join(speaker_dir, filename)
                
                # Get enrollment embedding
                enroll_embedding = exctract_vector_embedding(
                    enrollment_file,
                    self.model_name,
                    model,
                    feature_extractor
                )
                enroll_embedding = enroll_embedding.reshape(-1)  # Flatten to 1D
                enroll_embedding = enroll_embedding / np.linalg.norm(enroll_embedding)
                
                # Compute cosine similarity
                score = np.dot(test_embedding, enroll_embedding)
                
                if score > best_score:
                    best_score = score
                    best_file = enrollment_file
        
        if best_file is None:
            raise ValueError(f"No enrollment files found for speaker {test_speaker}")
        
        self.logger.info(f"✓ Closest enrollment: {os.path.basename(best_file)} (score: {best_score:.4f})")
        return best_file, best_score
    
    def compute_spectral_similarity(self,
                                   spec1: np.ndarray,
                                   spec2: np.ndarray,
                                   method: str = 'cosine') -> Tuple[np.ndarray, float]:
        """Compute spectral similarity between two spectrograms.
        
        Args:
            spec1 (np.ndarray): First spectrogram [n_mels, time_steps]
            spec2 (np.ndarray): Second spectrogram [n_mels, time_steps]
            method (str): Similarity method ('cosine', 'euclidean', 'correlation')
            
        Returns:
            tuple: (similarity_matrix, global_score)
                - similarity_matrix: Frame-wise similarity scores
                - global_score: Overall similarity (0-1)
        """
        # Normalize spectrograms
        spec1_norm = (spec1 - spec1.mean()) / (spec1.std() + 1e-10)
        spec2_norm = (spec2 - spec2.mean()) / (spec2.std() + 1e-10)
        
        # Handle different lengths by padding
        max_len = max(spec1_norm.shape[1], spec2_norm.shape[1])
        spec1_padded = np.pad(spec1_norm, ((0, 0), (0, max_len - spec1_norm.shape[1])))
        spec2_padded = np.pad(spec2_norm, ((0, 0), (0, max_len - spec2_norm.shape[1])))
        
        if method == 'cosine':
            # Compute cosine similarity frame by frame
            similarities = []
            for i in range(max_len):
                frame1 = spec1_padded[:, i]
                frame2 = spec2_padded[:, i]
                
                # Cosine similarity
                dot_product = np.dot(frame1, frame2)
                norms = np.linalg.norm(frame1) * np.linalg.norm(frame2)
                sim = dot_product / (norms + 1e-10)
                similarities.append(sim)
            
            similarity_matrix = np.array(similarities)
            
        elif method == 'correlation':
            # Use 2D correlation
            correlation = correlate2d(spec1_padded, spec2_padded, mode='same')
            similarity_matrix = correlation.mean(axis=0)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize to [0, 1]
        if similarity_matrix.min() < 0:
            similarity_matrix = (similarity_matrix - similarity_matrix.min()) / \
                               (similarity_matrix.max() - similarity_matrix.min() + 1e-10)
        
        global_score = similarity_matrix.mean()
        
        return similarity_matrix, global_score
    
    def identify_similar_regions(self,
                                similarity_scores: np.ndarray,
                                threshold: float = 0.5) -> Dict[str, List]:
        """Identify regions of high and low similarity.
        
        Args:
            similarity_scores (np.ndarray): Frame-wise similarity scores
            threshold (float): Threshold for "similar" vs "different"
            
        Returns:
            dict: Regions {'similar': [...], 'different': [...]}
        """
        similar_frames = np.where(similarity_scores >= threshold)[0]
        different_frames = np.where(similarity_scores < threshold)[0]
        
        # Group consecutive frames
        def group_consecutive(frames):
            groups = []
            if len(frames) == 0:
                return groups
            
            start = frames[0]
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] > 1:
                    groups.append((start, frames[i-1]))
                    start = frames[i]
            groups.append((start, frames[-1]))
            return groups
        
        return {
            'similar': group_consecutive(similar_frames),
            'different': group_consecutive(different_frames),
            'similarity_scores': similarity_scores
        }
    
    def plot_spectrogram_comparison(self,
                                   audio_file_1: str,
                                   audio_file_2: str,
                                   similarity_scores: Optional[np.ndarray] = None,
                                   title: str = "Speaker Comparison: Spectral Analysis",
                                   save_to: Optional[str] = None,
                                   sr: int = 16000) -> None:
        """Plot spectrograms with similarity overlay.
        
        Args:
            audio_file_1 (str): First audio file
            audio_file_2 (str): Second audio file
            similarity_scores (np.ndarray, optional): Frame-wise similarity
            title (str): Plot title
            save_to (str, optional): Path to save figure
            sr (int): Sample rate
        """
        # Load and process audio
        y1, _ = librosa.load(audio_file_1, sr=sr)
        y2, _ = librosa.load(audio_file_2, sr=sr)
        
        # Compute spectrograms
        D1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=128)
        S1_db = librosa.power_to_db(D1, ref=np.max)
        
        D2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=128)
        S2_db = librosa.power_to_db(D2, ref=np.max)
        
        # Create figure
        if similarity_scores is not None:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [2, 2, 1]})
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes = [axes[0], axes[1]]
        
        # Plot first spectrogram
        img1 = librosa.display.specshow(S1_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0])
        axes[0].set_title(f"Spectrogram 1: {os.path.basename(audio_file_1)}", 
                         fontsize=12, fontweight='bold')
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB', label='Power (dB)')
        
        # Plot second spectrogram
        img2 = librosa.display.specshow(S2_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
        axes[1].set_title(f"Spectrogram 2: {os.path.basename(audio_file_2)}", 
                         fontsize=12, fontweight='bold')
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB', label='Power (dB)')
        
        # Plot similarity scores if provided
        if similarity_scores is not None:
            times = librosa.frames_to_time(np.arange(len(similarity_scores)), sr=sr)
            
            # Color bars based on similarity
            colors = ['#2ecc71' if s > 0.5 else '#e74c3c' for s in similarity_scores]
            axes[2].bar(times, similarity_scores, width=(times[1]-times[0]), 
                       color=colors, edgecolor='black', linewidth=0.5)
            axes[2].set_ylabel('Similarity', fontsize=11, fontweight='bold')
            axes[2].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
            axes[2].set_title('Frame-wise Spectral Similarity', fontsize=12, fontweight='bold')
            axes[2].set_ylim([0, 1])
            axes[2].grid(True, alpha=0.3, axis='y')
            axes[2].axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ Comparison plot saved to {save_to}")
        
        plt.show()
    
    def analyze_and_compare(self,
                           test_file: str,
                           enrollment_dir: str,
                           title_prefix: str = "Speaker Comparison",
                           save_dir: Optional[str] = None,
                           sr: int = 16000) -> Dict:
        """Complete analysis: find closest enrollment and compare.
        
        Args:
            test_file (str): Test audio file
            enrollment_dir (str): Enrollment directory
            title_prefix (str): Prefix for plot titles
            save_dir (str, optional): Directory to save plots
            sr (int): Sample rate
            
        Returns:
            dict: Analysis results
            
        Example:
            >>> results = comparison.analyze_and_compare(
            ...     './test/SPK1_test.wav',
            ...     './enrollment/',
            ...     save_dir='./results/'
            ... )
            >>> print(f"Similarity: {results['global_similarity']:.2%}")
        """
        # Find closest match
        best_enrollment, match_score = self.find_closest_enrollment(test_file, enrollment_dir)
        
        # Compute similarity
        y_test, _ = librosa.load(test_file, sr=sr)
        y_enroll, _ = librosa.load(best_enrollment, sr=sr)
        
        D_test = librosa.feature.melspectrogram(y=y_test, sr=sr, n_mels=128)
        S_test = librosa.power_to_db(D_test, ref=np.max)
        
        D_enroll = librosa.feature.melspectrogram(y=y_enroll, sr=sr, n_mels=128)
        S_enroll = librosa.power_to_db(D_enroll, ref=np.max)
        
        similarity_scores, global_score = self.compute_spectral_similarity(S_test, S_enroll)
        
        # Identify regions
        regions = self.identify_similar_regions(similarity_scores, threshold=0.5)
        
        # Plot comparison
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'speaker_comparison.png')
        
        self.plot_spectrogram_comparison(
            test_file,
            best_enrollment,
            similarity_scores=similarity_scores,
            title=f"{title_prefix} (Similarity: {global_score:.2%})",
            save_to=save_path
        )
        
        return {
            'test_file': test_file,
            'closest_enrollment': best_enrollment,
            'embedding_similarity': match_score,
            'spectral_similarity': global_score,
            'similarity_scores': similarity_scores,
            'similar_regions': regions['similar'],
            'different_regions': regions['different']
        }


# Convenience functions
def compare_speakers(test_file: str,
                    enrollment_dir: str,
                    model_name: str = 'wavLM',
                    **kwargs) -> Dict:
    """Quick function to compare two speakers.
    
    Example:
        >>> results = compare_speakers(
        ...     './test/SPK1_test.wav',
        ...     './enrollment/'
        ... )
    """
    comparison = SpecificSpeakerComparison(model_name=model_name)
    return comparison.analyze_and_compare(test_file, enrollment_dir, **kwargs)


def plot_speaker_comparison(test_file: str,
                            enrollment_file: str,
                            **kwargs) -> None:
    """Plot comparison between a test and enrollment file.
    
    Example:
        >>> plot_speaker_comparison(
        ...     './test/SPK1_test.wav',
        ...     './enrollment/SPK1/SPK1_0001.wav'
        ... )
    """
    comparison = SpecificSpeakerComparison()
    
    y_test, sr = librosa.load(test_file, sr=16000)
    y_enroll, _ = librosa.load(enrollment_file, sr=16000)
    
    D_test = librosa.feature.melspectrogram(y=y_test, sr=sr, n_mels=128)
    S_test = librosa.power_to_db(D_test, ref=np.max)
    
    D_enroll = librosa.feature.melspectrogram(y=y_enroll, sr=sr, n_mels=128)
    S_enroll = librosa.power_to_db(D_enroll, ref=np.max)
    
    similarity_scores, global_score = comparison.compute_spectral_similarity(S_test, S_enroll)
    
    comparison.plot_spectrogram_comparison(
        test_file,
        enrollment_file,
        similarity_scores=similarity_scores,
        title=f"Speaker Comparison (Spectral Similarity: {global_score:.2%})",
        **kwargs
    )
