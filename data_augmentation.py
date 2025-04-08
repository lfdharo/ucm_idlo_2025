from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import soundfile as sf
from utils import find_files
import librosa
import os

class DataAugmentation:
    def __init__(self, main_path: str):
        self.main_path = main_path

        # Options for data augmentation available at https://github.com/iver56/audiomentations

        self.augment_gaussianNoise_fnt = Compose([
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=1.0),
        ])

        self.augment_TimeStretch_fnt = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        ])

        self.augment_PitchShift_fnt = Compose([
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
        ])

        self.augment_Shift_fnt = Compose([
            Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
        ])

        self.augment_all_fnt = Compose([
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
        ])

    def augment_gaussianNoise(self, signal, fs, user_name, wav_file):
        # Augment/transform/perturb the audio data
        augmented_samples = self.augment_gaussianNoise_fnt(samples=signal, sample_rate=fs)
        sf.write(os.path.join(user_name, wav_file) + '_gaussian.wav', augmented_samples, fs)
    
    def augment_TimeStretch(self, signal, fs, user_name, wav_file):
        augmented_samples = self.augment_TimeStretch_fnt(samples=signal, sample_rate=fs)
        sf.write(os.path.join(user_name, wav_file) + '_timeStretch.wav', augmented_samples, fs)

    def augment_PitchShift(self, signal, fs, user_name, wav_file):
        augmented_samples = self.augment_PitchShift_fnt(samples=signal, sample_rate=fs)
        sf.write(os.path.join(user_name, wav_file) + '_pitchShift.wav', augmented_samples, fs)

    def augment_Shift(self, signal, fs, user_name, wav_file):
        augmented_samples = self.augment_Shift_fnt(samples=signal, sample_rate=fs)
        sf.write(os.path.join(user_name, wav_file) + '_shift.wav', augmented_samples, fs)

    def augment_all(self, signal, fs, user_name, wav_file):
        augmented_samples = self.augment_all_fnt(samples=signal, sample_rate=fs)
        sf.write(os.path.join(user_name, wav_file) + '_all.wav', augmented_samples, fs)

    def augment_data(self, folder: str, type: str, speaker: str = None):
        list_wav_files = find_files(os.path.join(self.main_path, folder), speaker)

        for file in list_wav_files:
            if 'gaussian' in file or 'timeStretch' in file or \
            'pitchShift' in file or 'shift' in file or 'all' in file:
                continue
            print('Processing file {}'.format(file))
            user_name = os.path.dirname(file)
            wav_file, _ = os.path.splitext(os.path.basename(file))
            signal, fs = librosa.load(file)
            
            if type == 'gaussianNoise':
                self.augment_gaussianNoise(signal, fs, user_name, wav_file)
            elif type == 'timeStretch': 
                self.augment_TimeStretch(signal, fs, user_name, wav_file)
            elif type == 'pitchShift':
                self.augment_PitchShift(signal, fs, user_name, wav_file)
            elif type == 'shift':    
                self.augment_Shift(signal, fs, user_name, wav_file)
            elif type == 'all':
                self.augment_all(signal, fs, user_name, wav_file)
            else:
                raise ValueError('Invalid type. Options are: gaussianNoise, timeStretch, pitchShift, shift, all')