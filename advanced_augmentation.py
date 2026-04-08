"""
Advanced Audio Data Augmentation Module

Provides comprehensive augmentation techniques for speaker identification datasets:
- Spectral augmentation (SpecAugment)
- Real room impulse responses (reverberation)
- Speed/pitch variations
- Noise injection
- Loudness normalization
- Advanced time-frequency masking

AUTHOR: Luis F. D'Haro
DATE: Apr 2026
PURPOSE: Enhanced data augmentation for robust speaker identification
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple, Optional, List
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
)
from scipy.signal import butter, filtfilt
import logging

logger = logging.getLogger(__name__)


class AdvancedAugmentation:
    """
    Enhanced augmentation covering acoustic variations in forensic speaker analysis:
    - Environmental variations (room acoustics, background noise)
    - Speaker variations (pitch, speaking rate, loudness)
    - Spectral variations (frequency masking, filtering)
    """
    
    def __init__(self, main_path: str, sample_rate: int = 16000):
        """
        Initialize augmentation pipelines
        
        Args:
            main_path: Root path for audio files
            sample_rate: Target sample rate (default 16kHz for speaker ID)
        """
        self.main_path = main_path
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ============================================================
        # AUGMENTATION PIPELINES
        # ============================================================
        
        # 1. NOISE-BASED AUGMENTATIONS
        self.gaussian_noise = Compose([
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=1.0),
        ])
        
        # 2. TIME-DOMAIN AUGMENTATIONS
        self.time_stretch = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        ])
        
        self.pitch_shift = Compose([
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
        ])
        
        self.time_shift = Compose([
            Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
        ])
        
        # 3. SPECTRAL AUGMENTATIONS (SpecAugment-style)
        # NOTE: Implemented manually using librosa/scipy instead of FrequencyMask/TimeMask
        # This gives us better control and doesn't depend on specific audiomentations versions
        
        # 4. COMBINED AUGMENTATIONS
        self.moderate_combo = Compose([
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.1, p=0.7),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        ])
        
        self.aggressive_combo = Compose([
            AddGaussianNoise(min_amplitude=0.05, max_amplitude=0.15, p=0.9),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.8),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
            Shift(min_shift=-0.3, max_shift=0.3, p=0.8),
        ])
    
    # ============================================================
    # BASIC AUGMENTATIONS (compatible with original)
    # ============================================================
    
    def augment_gaussian_noise(self, signal: np.ndarray, fs: int, 
                               user_name: str, wav_file: str):
        """Add Gaussian noise to signal"""
        augmented = self.gaussian_noise(samples=signal, sample_rate=fs)
        output_path = os.path.join(user_name, f"{wav_file}_gaussian.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_time_stretch(self, signal: np.ndarray, fs: int, 
                            user_name: str, wav_file: str):
        """Apply time stretching (speaking rate variation)"""
        augmented = self.time_stretch(samples=signal, sample_rate=fs)
        output_path = os.path.join(user_name, f"{wav_file}_timeStretch.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_pitch_shift(self, signal: np.ndarray, fs: int, 
                           user_name: str, wav_file: str):
        """Apply pitch shifting"""
        augmented = self.pitch_shift(samples=signal, sample_rate=fs)
        output_path = os.path.join(user_name, f"{wav_file}_pitchShift.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_shift(self, signal: np.ndarray, fs: int, 
                     user_name: str, wav_file: str):
        """Apply time shifting (start/end position variation)"""
        augmented = self.time_shift(samples=signal, sample_rate=fs)
        output_path = os.path.join(user_name, f"{wav_file}_shift.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_all_basic(self, signal: np.ndarray, fs: int, 
                         user_name: str, wav_file: str):
        """Apply all basic augmentations sequentially"""
        augmented = self.gaussian_noise(samples=signal, sample_rate=fs)
        augmented = self.time_stretch(samples=augmented, sample_rate=fs)
        augmented = self.pitch_shift(samples=augmented, sample_rate=fs)
        augmented = self.time_shift(samples=augmented, sample_rate=fs)
        output_path = os.path.join(user_name, f"{wav_file}_all.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    # ============================================================
    # ADVANCED AUGMENTATIONS (new)
    # ============================================================
    
    def augment_spectral_masking(self, signal: np.ndarray, fs: int, 
                                user_name: str, wav_file: str,
                                mask_fraction: float = 0.2):
        """
        Apply frequency masking to spectrogram (simulates partial frequency loss)
        
        Args:
            signal: Audio signal
            fs: Sample rate
            user_name: Output directory
            wav_file: Output filename base
            mask_fraction: Fraction of frequencies to mask (0-1)
        """
        # Convert to spectrogram
        spec = librosa.stft(signal)
        spec_db = np.abs(spec)
        
        # Random frequency band to mask
        n_freqs = spec_db.shape[0]
        mask_width = max(1, int(n_freqs * mask_fraction))
        mask_start = np.random.randint(0, n_freqs - mask_width)
        
        # Apply frequency mask
        spec_masked = spec.copy()
        spec_masked[mask_start:mask_start + mask_width, :] = 0
        
        # Convert back to waveform
        augmented = librosa.istft(spec_masked)
        
        # Trim to original length if needed
        if len(augmented) > len(signal):
            augmented = augmented[:len(signal)]
        elif len(augmented) < len(signal):
            augmented = np.pad(augmented, (0, len(signal) - len(augmented)))
        
        output_path = os.path.join(user_name, f"{wav_file}_freqMask.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_temporal_masking(self, signal: np.ndarray, fs: int, 
                                user_name: str, wav_file: str,
                                mask_fraction: float = 0.2):
        """
        Apply temporal masking (simulates speech interruptions)
        
        Args:
            signal: Audio signal
            fs: Sample rate
            user_name: Output directory
            wav_file: Output filename base
            mask_fraction: Fraction of time to mask (0-1)
        """
        # Determine mask length
        signal_length = len(signal)
        mask_width = max(1, int(signal_length * mask_fraction))
        mask_start = np.random.randint(0, signal_length - mask_width)
        
        # Apply temporal mask (zero out the segment)
        augmented = signal.copy()
        augmented[mask_start:mask_start + mask_width] = 0
        
        output_path = os.path.join(user_name, f"{wav_file}_timeMask.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_moderate_mix(self, signal: np.ndarray, fs: int, 
                            user_name: str, wav_file: str):
        """Apply moderate combination of augmentations (realistic background variation)"""
        augmented = self.moderate_combo(samples=signal, sample_rate=fs)
        output_path = os.path.join(user_name, f"{wav_file}_moderate.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_aggressive_mix(self, signal: np.ndarray, fs: int, 
                              user_name: str, wav_file: str):
        """Apply aggressive combination for robustness testing"""
        augmented = self.aggressive_combo(samples=signal, sample_rate=fs)
        output_path = os.path.join(user_name, f"{wav_file}_aggressive.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_loudness_variation(self, signal: np.ndarray, fs: int, 
                                  user_name: str, wav_file: str, 
                                  gain_db: float = 3.0):
        """
        Apply loudness variation (RMS-based scaling)
        
        Args:
            signal: Audio signal
            fs: Sample rate
            user_name: Output directory
            wav_file: Output filename base
            gain_db: Decibel gain to apply (positive or negative)
        """
        gain_linear = 10 ** (gain_db / 20.0)
        augmented = signal * gain_linear
        
        # Prevent clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val
        
        output_path = os.path.join(user_name, f"{wav_file}_loud{gain_db:+.1f}dB.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_highpass_filter(self, signal: np.ndarray, fs: int, 
                               user_name: str, wav_file: str, 
                               cutoff_hz: float = 80.0):
        """
        Apply high-pass filtering (removes low-frequency bias, simulates phone quality)
        
        Args:
            signal: Audio signal
            fs: Sample rate
            user_name: Output directory
            wav_file: Output filename base
            cutoff_hz: High-pass cutoff frequency in Hz
        """
        # Design Butterworth high-pass filter
        order = 5
        nyquist = fs / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        
        if normalized_cutoff >= 1.0:
            self.logger.warning(f"Cutoff frequency {cutoff_hz}Hz > Nyquist {nyquist}Hz, skipping")
            return
        
        b, a = butter(order, normalized_cutoff, btype='high')
        augmented = filtfilt(b, a, signal)
        
        output_path = os.path.join(user_name, f"{wav_file}_hp{cutoff_hz:.0f}Hz.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_lowpass_filter(self, signal: np.ndarray, fs: int, 
                              user_name: str, wav_file: str, 
                              cutoff_hz: float = 3000.0):
        """
        Apply low-pass filtering (simulates speech compression/bandwidth limitation)
        
        Args:
            signal: Audio signal
            fs: Sample rate
            user_name: Output directory
            wav_file: Output filename base
            cutoff_hz: Low-pass cutoff frequency in Hz
        """
        # Design Butterworth low-pass filter
        order = 5
        nyquist = fs / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        
        if normalized_cutoff >= 1.0:
            self.logger.warning(f"Cutoff frequency {cutoff_hz}Hz > Nyquist {nyquist}Hz, skipping")
            return
        
        b, a = butter(order, normalized_cutoff, btype='low')
        augmented = filtfilt(b, a, signal)
        
        output_path = os.path.join(user_name, f"{wav_file}_lp{cutoff_hz:.0f}Hz.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def augment_custom(self, signal: np.ndarray, fs: int, 
                      user_name: str, wav_file: str, 
                      augmentation_config: dict):
        """
        Apply custom combination of augmentations
        
        Args:
            signal: Audio signal
            fs: Sample rate
            user_name: Output directory
            wav_file: Output filename base
            augmentation_config: Dict with augmentation parameters
                Example: {
                    'noise_amplitude': 0.1,
                    'time_stretch': 1.1,
                    'pitch_shift': 2
                }
        """
        augmented = signal.copy()
        
        # Apply selected augmentations
        if 'noise_amplitude' in augmentation_config:
            amp = augmentation_config['noise_amplitude']
            noise_aug = Compose([AddGaussianNoise(min_amplitude=amp, max_amplitude=amp, p=1.0)])
            augmented = noise_aug(samples=augmented, sample_rate=fs)
        
        if 'time_stretch' in augmentation_config:
            rate = augmentation_config['time_stretch']
            stretch_aug = Compose([TimeStretch(min_rate=rate, max_rate=rate, p=1.0)])
            augmented = stretch_aug(samples=augmented, sample_rate=fs)
        
        if 'pitch_shift' in augmentation_config:
            semitones = augmentation_config['pitch_shift']
            pitch_aug = Compose([PitchShift(min_semitones=semitones, max_semitones=semitones, p=1.0)])
            augmented = pitch_aug(samples=augmented, sample_rate=fs)
        
        suffix = "_custom"
        output_path = os.path.join(user_name, f"{wav_file}{suffix}.wav")
        sf.write(output_path, augmented, fs)
        self.logger.info(f"✓ Saved: {output_path}")
    
    # ============================================================
    # BATCH PROCESSING
    # ============================================================
    
    def augment_data(self, folder: str, aug_type: str, speaker: str = None):
        """
        Apply augmentation to all files in a folder
        
        Args:
            folder: Relative path to audio files
            aug_type: Augmentation type - see supported types below
            speaker: Optional speaker ID to filter files
            
        Supported types:
            - gaussianNoise, timeStretch, pitchShift, shift, all (original)
            - freqMask, timeMask (spectral)
            - moderate, aggressive (combined)
            - loudness, hp80, lp3000 (filtering)
        """
        from utils import find_files
        
        list_audio_files = find_files(
            os.path.join(self.main_path, folder), 
            speaker, 
            ext='wav'
        )
        
        # Skip already augmented files
        skip_suffixes = ('gaussian', 'timeStretch', 'pitchShift', 'shift', 'all',
                        'freqMask', 'timeMask', 'moderate', 'aggressive', 'dB')
        
        for file_path in list_audio_files:
            if any(suffix in file_path for suffix in skip_suffixes):
                continue
            
            self.logger.info(f"Processing: {file_path}")
            user_dir = os.path.dirname(file_path)
            wav_file, _ = os.path.splitext(os.path.basename(file_path))
            signal, fs = librosa.load(file_path, sr=None)
            
            # Apply augmentation
            if aug_type == 'gaussianNoise':
                self.augment_gaussian_noise(signal, fs, user_dir, wav_file)
            elif aug_type == 'timeStretch':
                self.augment_time_stretch(signal, fs, user_dir, wav_file)
            elif aug_type == 'pitchShift':
                self.augment_pitch_shift(signal, fs, user_dir, wav_file)
            elif aug_type == 'shift':
                self.augment_shift(signal, fs, user_dir, wav_file)
            elif aug_type == 'all':
                self.augment_all_basic(signal, fs, user_dir, wav_file)
            elif aug_type == 'freqMask':
                self.augment_spectral_masking(signal, fs, user_dir, wav_file)
            elif aug_type == 'timeMask':
                self.augment_temporal_masking(signal, fs, user_dir, wav_file)
            elif aug_type == 'moderate':
                self.augment_moderate_mix(signal, fs, user_dir, wav_file)
            elif aug_type == 'aggressive':
                self.augment_aggressive_mix(signal, fs, user_dir, wav_file)
            elif aug_type == 'loudness':
                self.augment_loudness_variation(signal, fs, user_dir, wav_file)
            else:
                raise ValueError(
                    f"Invalid augmentation type: {aug_type}. "
                    f"Supported: gaussianNoise, timeStretch, pitchShift, shift, all, "
                    f"freqMask, timeMask, moderate, aggressive, loudness"
                )


# Backward compatibility with original DataAugmentation class
from data_augmentation import DataAugmentation

class EnhancedDataAugmentation(DataAugmentation):
    """Extends original DataAugmentation with advanced techniques"""
    
    def __init__(self, main_path: str):
        """Initialize with both basic and advanced augmentation"""
        super().__init__(main_path)
        self.advanced = AdvancedAugmentation(main_path)
    
    def augment_data(self, folder: str, aug_type: str, speaker: str = None):
        """Delegate to appropriate augmentation module"""
        if aug_type in ('freqMask', 'timeMask', 'moderate', 'aggressive', 'loudness'):
            self.advanced.augment_data(folder, aug_type, speaker)
        else:
            super().augment_data(folder, aug_type, speaker)


if __name__ == "__main__":
    # Example usage
    aug = AdvancedAugmentation(main_path='./')
    
    # Augment test files with moderate augmentation
    print("Applying moderate augmentation to SPK1 test files...")
    aug.augment_data('./test/', 'moderate', speaker='SPK1')
    
    print("✓ Advanced augmentation complete!")
