"""
Simple, Fast TTS Fallback for Testing and Development

For when heavy TTS models are slow or unavailable.
Generates synthetic speech using:
1. espeak (system-based, no models needed)
2. pyttsx3 (cross-platform, lightweight)
3. Simple noise-based audio for testing

AUTHOR: Luis F. D'Haro
DATE: Apr 2026
PURPOSE: Lightweight alternative when heavy TTS models are unavailable
"""

import os
import logging
import numpy as np
import soundfile as sf
from typing import Optional

logger = logging.getLogger(__name__)


class SimpleTTS:
    """
    Lightweight TTS using system speech engines
    """
    
    def __init__(self, backend: str = 'auto', device: Optional[str] = None):
        """
        Initialize Simple TTS
        
        Args:
            backend: 'espeak', 'pyttsx3', 'mock', or 'auto' (try in order)
            device: Ignored (for API compatibility)
        """
        self.backend = backend
        self.sample_rate = 16000
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available_backend = None
        self._select_backend()
    
    def _select_backend(self):
        """Select available backend"""
        backends_to_try = {
            'espeak': self._check_espeak,
            'pyttsx3': self._check_pyttsx3,
            'mock': self._check_mock,
            'qwen3tts': self._check_qwen3tts,
        }
        
        if self.backend == 'auto':
            for name, check in [('espeak', backends_to_try['espeak']),
                               ('pyttsx3', backends_to_try['pyttsx3']),
                               ('mock', backends_to_try['mock']),
                               ('qwen3tts', backends_to_try['qwen3tts'])]:
                if check():
                    self.available_backend = name
                    self.logger.info(f"SimpleTTS using: {name}")
                    return
        else:
            if self.backend in backends_to_try and backends_to_try[self.backend]():
                self.available_backend = self.backend
                self.logger.info(f"SimpleTTS using: {self.backend}")
                return
        
        # Fallback to mock
        self.available_backend = 'mock'
        self.logger.warning("No TTS backend available, using mock (silent audio)")
    
    def _check_espeak(self) -> bool:
        """Check if espeak is available"""
        try:
            import subprocess
            result = subprocess.run(['espeak', '--version'],
                                   capture_output=True, timeout=1)
            return result.returncode == 0
        except:
            return False
    
    def _check_pyttsx3(self) -> bool:
        """Check if pyttsx3 is available"""
        try:
            import pyttsx3
            return True
        except ImportError:
            return False
    
    def _check_qwen3tts(self) -> bool:
        try:
            from qwen_tts import Qwen3TTSModel
            import torch

            # Load the model            
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if device.startswith('cuda') else "sdpa",
            )            
            return True
        except ImportError:
            return False
    
    def _check_mock(self) -> bool:
        """Mock backend is always available"""
        return True
    
    def generate_speech(self, text: str, output_file: str,
                       language: str = 'es', speed: float = 1.0, **kwargs) -> bool:
        """
        Generate speech from text
        
        Args:
            text: Text to synthesize
            output_file: Output WAV file path
            language: Language code (e.g., 'es', 'en')
            speed: Speech rate (1.0 = normal)
            
        Returns:
            True if successful
        """
        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            
            if self.available_backend == 'espeak':
                return self._generate_espeak(text, output_file, language, speed)
            elif self.available_backend == 'pyttsx3':
                return self._generate_pyttsx3(text, output_file)
            elif self.available_backend == 'qwen3tts':
                return self._generate_qwen3tts(text, output_file, language, speed, **kwargs)
            else:  # mock
                return self._generate_mock(text, output_file)
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return False
    
    def _generate_espeak(self, text: str, output_file: str,
                        language: str, speed: float) -> bool:
        """Generate using espeak command-line tool"""
        try:
            import subprocess
            
            # Format language code for espeak
            lang_map = {'es': 'es-es',  'en': 'en', 'fr': 'fr', 'de': 'de'}
            espeak_lang = lang_map.get(language[:2], 'en')
            
            # espeak command
            rate = int(150 * speed)  # Words per minute
            cmd = [
                'espeak',
                f'-v{espeak_lang}',
                f'-s{rate}',
                f'-w{output_file}',
                text
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            success = result.returncode == 0
            
            if success:
                self.logger.debug(f"Generated using espeak: {output_file}")
            else:
                self.logger.error(f"espeak error: {result.stderr.decode('utf-8', errors='ignore')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"espeak generation failed: {e}")
            return False
    
    def _generate_pyttsx3(self, text: str, output_file: str) -> bool:
        """Generate using pyttsx3"""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
            success = os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
            if success:
                self.logger.debug(f"Generated using pyttsx3: {output_file}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"pyttsx3 generation failed: {e}")
            return False
    
    def _generate_qwen3tts(self, text: str, output_file: str, language: str, speed: float, ref_audio: str, ref_text: str) -> bool:
        """Generate using Qwen3TTSModel (if available)"""
        try:
            from qwen_tts import Qwen3TTSModel
            import soundfile as sf
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,  # Using input text as reference text for simplicity
            )
            
            sf.write(output_file, wavs[0], sr)
            self.logger.debug(f"Generated using Qwen3TTS: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Qwen3TTS generation failed: {e}")
            return False
    
    def _generate_mock(self, text: str, output_file: str) -> bool:
        """Generate mock audio for testing (silence with modulation)"""
        try:
            # Create synthetic audio with noise based on text length
            duration = max(1.0, len(text) / 100)  # ~1 second per ~100 chars
            samples = int(duration * self.sample_rate)
            
            # Generate filtered noise
            noise = np.random.normal(0, 0.1, samples).astype(np.float32)
            
            # Add modulation based on text (simple pseudo-speech)
            t = np.linspace(0, duration, samples, dtype=np.float32)
            freq = 200 + 100 * np.sin(2 * np.pi * t / duration)
            modulation = np.sin(2 * np.pi * freq * t / self.sample_rate)
            
            audio = noise * np.abs(modulation) * 0.5
            
            # Soft amplitude envelope (Hann window)
            envelope = 0.5 * (1 - np.cos(2 * np.pi * np.arange(samples) / samples))
            audio = audio * envelope
            
            sf.write(output_file, audio, self.sample_rate)
            
            self.logger.debug(f"Generated mock audio: {output_file} ({duration:.1f}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Mock generation failed: {e}")
            return False


if __name__ == '__main__':
    # Test
    tts = SimpleTTS(backend='qwen3tts')
    print(f"Using backend: {tts.available_backend}")
    result = tts.generate_speech(text="Hola, ¿cómo estás? pasaba por aquí.", output_file="/tmp/test1.wav", language='spanish', speed=1.0, ref_audio="./enrollment/SPK1/SPK1_0001.wav", ref_text="esta es una prueba para ver si funcional el programa")
    print(f"Result: {result}")
