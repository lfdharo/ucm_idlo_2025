"""
TTS Wrapper: Qwen3TTS for Voice Cloning + CoquiTTS Fallback

Supports:
1. Qwen3TTS: High-quality voice cloning (requires ref_audio + ref_text)
   - Uses prompt caching for efficient multi-generation with same reference
   - Recommended for cloning applications
   
2. CoquiTTS: Flexible TTS for both regular synthesis and cloning
   - Can generate speech from text alone (no reference needed)
   - Fallback when Qwen3TTS not available or for non-cloning use cases

AUTHOR: Luis F. D'Haro
DATE: Apr 2026
PURPOSE: Voice cloning and TTS synthesis for speaker identification
"""

import os
import logging
import numpy as np
import soundfile as sf
from typing import Optional, List, Tuple
import torch

logger = logging.getLogger(__name__)


class SimpleTTS:
    """
    TTS wrapper supporting:
    - Qwen3TTS: Voice cloning with prompt caching
    - CoquiTTS: Fallback for both TTS and cloning
    """
    
    def __init__(self, backend: str = 'auto', device: Optional[str] = None,
                 x_vector_only_mode: bool = True):
        """
        Initialize TTS system
        
        Args:
            backend: 'qwen3tts', 'coqui', or 'auto' (try Qwen3 first, fallback to Coqui)
            device: GPU device for inference (e.g., 'cuda:0', 'cpu')
            x_vector_only_mode: If True, uses only speaker embedding (faster, no text concat).
                                If False, uses ICL mode (better quality but may concat text).
        """
        self.backend = backend
        self.x_vector_only_mode = x_vector_only_mode  # NEW: control ICL vs x-vector only
        self.backend = backend
        self.sample_rate = 16000
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.available_backend = None
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Model instances
        self.qwen_model = None
        self.qwen_voice_prompt_cache = {}  # Cache: (ref_audio_hash) -> prompt_items
        self.coqui_tts = None
        
        self._select_backend()
    
    def _select_backend(self):
        """Initialize and select available backend"""
        backends_to_try = [
            ('qwen3tts', self._init_qwen3tts),
            ('coqui', self._init_coqui),
        ]
        
        if self.backend == 'auto':
            for name, init_func in backends_to_try:
                if init_func():
                    self.available_backend = name
                    self.logger.info(f"SimpleTTS initialized: {name}")
                    return
            self.logger.error("No TTS backend available!")
        else:
            # Try specific backend
            backend_map = {'qwen3tts': self._init_qwen3tts, 'coqui': self._init_coqui}
            if self.backend in backend_map and backend_map[self.backend]():
                self.available_backend = self.backend
                self.logger.info(f"SimpleTTS initialized: {self.backend}")
                return
            else:
                self.logger.error(f"Failed to initialize {self.backend} backend")
    
    def _init_qwen3tts(self) -> bool:
        """Initialize Qwen3TTS for voice cloning with Flash Attention optimization"""
        try:
            from qwen_tts import Qwen3TTSModel
            
            self.logger.info(f"Loading Qwen3TTS on device: {self.device}")
            self.logger.info(f"Using x_vector_only_mode: {self.x_vector_only_mode}")
            
            # Try Flash Attention 2 for better performance
            attn_impl = "flash_attention_2"
            try:
                self.qwen_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    device_map=self.device,
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                )
            except Exception as attn_err:
                self.logger.warning(f"Flash Attention 2 not available: {attn_err}, falling back to default")
                self.qwen_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    device_map=self.device,
                    dtype=torch.bfloat16,
                )
            
            self.logger.debug("✓ Qwen3TTS loaded successfully")
            return True
        except Exception as e:
            self.logger.debug(f"Qwen3TTS initialization failed: {e}")
            return False
    
    def _init_coqui(self) -> bool:
        """Initialize CoquiTTS for fallback TTS"""
        try:
            from TTS.api import TTS
            
            self.logger.info(f"Loading CoquiTTS on device: {self.device}")
            # Use multilingual model that supports voice cloning
            self.coqui_tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True,
                gpu=(self.device.startswith('cuda'))
            )
            self.logger.debug("✓ CoquiTTS loaded successfully")
            return True
        except Exception as e:
            self.logger.debug(f"CoquiTTS initialization failed: {e}")
            return False
    
    def generate_speech(self, text: str, output_file: str,
                       language: str = 'es', speed: float = 1.0, 
                       ref_audio: Optional[str] = None, 
                       ref_text: Optional[str] = None, 
                       x_vector_only: Optional[bool] = None, **kwargs) -> bool:
        """
        Generate speech from text with optional voice cloning
        
        Args:
            text: Text to synthesize
            output_file: Output WAV file path
            language: Language code (e.g., 'es', 'en')
            speed: Speech speed multiplier (1.0 = normal)
            ref_audio: Optional reference audio file for voice cloning
            ref_text: Optional reference text for voice cloning (required for ICL mode)
            x_vector_only: Override default x_vector_only_mode for this call
            **kwargs: Additional arguments (ignored)
            
        Note:
            - x_vector_only_mode=True: Faster, uses only speaker embedding. No text concat.
            - x_vector_only_mode=False: Better quality but may concatenate ref_text with new text.
        """
        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            
            # Voice cloning request - only ref_audio is required
            if ref_audio:
                # Use per-call override or default
                use_xvec = x_vector_only if x_vector_only is not None else self.x_vector_only_mode
                
                if self.available_backend == 'qwen3tts' and self.qwen_model:
                    return self._generate_qwen3_clone(
                        text=text,
                        output_file=output_file,
                        language=language,
                        ref_audio=ref_audio,
                        ref_text=ref_text if not use_xvec else None,  # Only pass ref_text if not x-vector only
                        x_vector_only_mode=use_xvec
                    )
                elif self.available_backend == 'coqui' and self.coqui_tts:
                    return self._generate_coqui_clone(
                        text=text,
                        output_file=output_file,
                        language=language,
                        ref_audio=ref_audio,
                    )
                else:
                    self.logger.error("No cloning-capable backend available")
                    return False
            
            # Regular TTS (no cloning)
            else:
                if self.available_backend == 'coqui' and self.coqui_tts:
                    return self._generate_coqui_tts(
                        text=text,
                        output_file=output_file,
                        language=language,
                        speed=speed
                    )
                else:
                    self.logger.error(f"Backend {self.available_backend} cannot generate speech without cloning reference")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Speech generation failed: {e}")
            return False
    
    # ============================================================
    # QWEN3TTS: Voice Cloning with Prompt Caching
    # ============================================================
    
    def _generate_qwen3_clone(self, text: str, output_file: str,
                             language: str, ref_audio: str, ref_text: Optional[str] = None,
                             x_vector_only_mode: bool = True) -> bool:
        """
        Generate cloned voice using Qwen3TTS with prompt caching optimization
        
        For efficiency, creates and caches voice prompts to avoid recomputing
        the reference audio features for multiple generations.
        
        Args:
            text: Text to synthesize
            output_file: Output file path
            language: Language code
            ref_audio: Reference audio file for voice cloning
            ref_text: Text corresponding to reference audio (required if x_vector_only_mode=False)
            x_vector_only_mode: If True, uses only speaker embedding (faster, no text concat).
                               If False, uses ICL mode (better quality but may concat text).
        
        Note:
            - x_vector_only_mode=True: Recommended for spoofing (faster, avoids text concat)
            - x_vector_only_mode=False: Higher quality but may concatenate ref_text with new text
        """
        try:
            import hashlib
            
            # Create cache key from reference audio file + x_vector_only_mode
            with open(ref_audio, 'rb') as f:
                ref_audio_hash = hashlib.md5(f.read()).hexdigest()
            
            # Include x_vector_only_mode in cache key
            cache_key = f"{ref_audio_hash}_{x_vector_only_mode}"
            
            # Check if we already have this voice prompt cached
            if cache_key not in self.qwen_voice_prompt_cache:
                self.logger.debug(f"Creating voice prompt for {ref_audio} (x_vector_only={x_vector_only_mode})...")
                # Build voice prompt once and cache it
                voice_prompt = self.qwen_model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
                self.qwen_voice_prompt_cache[cache_key] = voice_prompt
                self.logger.debug("Voice prompt cached for future generations")
            else:
                self.logger.debug("Using cached voice prompt")
            
            # Generate using cached prompt
            mode_desc = "x-vector only" if x_vector_only_mode else "ICL"
            self.logger.debug(f"Generating cloned speech with Qwen3TTS ({mode_desc}) from {ref_audio} with text: {text} to {output_file}")
            voice_prompt = self.qwen_voice_prompt_cache[cache_key]
            wavs, sr = self.qwen_model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_prompt,
            )
            
            # Save output
            sf.write(output_file, wavs[0], sr)
            self.logger.debug(f"✓ Generated cloned speech: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Qwen3TTS cloning failed: {e}")
            return False
    
    # ============================================================
    # COQUI TTS: Fallback for both TTS and Cloning
    # ============================================================
    
    def _generate_coqui_tts(self, text: str, output_file: str,
                           language: str, speed: float = 1.0) -> bool:
        """
        Generate speech using CoquiTTS (regular TTS without cloning)
        
        Args:
            text: Text to synthesize
            output_file: Output file path
            language: Language code
            speed: Speech speed multiplier
            
        Returns:
            True if successful
        """
        try:
            self.logger.debug(f"Generating speech with CoquiTTS (language: {language})")
            
            # CoquiTTS returns (wav, sample_rate)
            wav = self.coqui_tts.tts(
                text=text,
                language=language,
                speaker_wav=None,  # No cloning
            )
            
            # Apply speed if needed
            if speed != 1.0:
                wav = self._apply_speed(wav, speed)
            
            sf.write(output_file, wav, self.coqui_tts.synthesizer.output_sample_rate)
            self.logger.debug(f"✓ Generated TTS speech: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"CoquiTTS generation failed: {e}")
            return False
    
    def _generate_coqui_clone(self, text: str, output_file: str,
                             language: str, ref_audio: str) -> bool:
        """
        Generate cloned voice using CoquiTTS
        
        CoquiTTS (XTTS v2) can perform voice cloning by using a reference audio
        
        Args:
            text: Text to synthesize
            output_file: Output file path
            language: Language code
            ref_audio: Reference audio file for voice cloning
            
        Returns:
            True if successful
        """
        try:
            self.logger.debug(f"Generating cloned speech with CoquiTTS from {ref_audio}")
            
            # CoquiTTS cloning using speaker_wav
            wav = self.coqui_tts.tts(
                text=text,
                language=language,
                speaker_wav=ref_audio,  # Use reference audio for cloning
            )
            
            sf.write(output_file, wav, self.coqui_tts.synthesizer.output_sample_rate)
            self.logger.debug(f"✓ Generated cloned speech: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"CoquiTTS cloning failed: {e}")
            return False
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Apply pitch-preserving speed change to audio
        
        Args:
            audio: Audio samples
            speed: Speed multiplier (1.0 = normal)
            
        Returns:
            Speed-adjusted audio
        """
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except Exception as e:
            self.logger.warning(f"Speed adjustment failed: {e}, using original speed")
            return audio
    
    def clear_voice_cache(self):
        """Clear cached voice prompts to free memory"""
        self.qwen_voice_prompt_cache.clear()
        self.logger.info("Voice prompt cache cleared")
