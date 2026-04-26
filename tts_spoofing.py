"""
TTS-Based Speaker Spoofing Module

Generates synthetic speech for evaluating speaker identification robustness using:
- Qwen3TTS: High-quality voice cloning with prompt caching optimization
- CoquiTTS: Flexible fallback supporting both TTS and cloning

Features:
- Controlled spoofing attempts with multiple text templates (Spanish)
- Voice cloning: Generate synthetic speaker attempts matching reference voice
- Batch generation: Create multiple variations for robustness testing
- Comparison metrics: Real vs Synthetic speaker verification rates

Forensic Use Case:
Students can use this module to test if their speaker ID system correctly
rejects or accepts synthetic attempts of an enrolled speaker.

AUTHOR: Luis F. D'Haro
DATE: Apr 2026
PURPOSE: TTS-based spoofing evaluation for speaker identification
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import librosa
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


class TTSSpoofingGenerator:
    """
    Generate synthetic speech attempts to spoof speaker identification system.
    
    Provides:
    1. Voice cloning with script variation
    2. Batch synthetic speech generation
    3. Spoofing difficulty levels
    4. Comparison between real and synthetic speaker verification
    
    Spanish Text Templates:
    - Professional: Formal speaker identification phrases
    - Casual: Conversational phrases
    - Numbers: Digit sequences
    - Mixed: Variety of phonetically diverse sentences
    """
    
    def __init__(self, model: str = 'simple', use_gpu: bool = True, x_vector_only: bool = True):
        """
        Initialize TTS for spoofing generation
        
        Args:
            model: TTS model to use:
                - 'simple': Auto-select (tries Qwen3TTS for cloning, CoquiTTS for TTS)
                - 'qwen3tts': Qwen3TTS voice cloning (GPU required)
                - 'coqui': CoquiTTS (supports both TTS and cloning)
            use_gpu: Use GPU if available (recommended)
            x_vector_only: If True, uses only speaker embedding (faster, no text concat).
                          If False, uses ICL mode (better quality but may concat text).
                          Recommended: True for spoofing to avoid text concatenation.
        """
        self.tts = None
        self.tts_model = model
        self.use_gpu = use_gpu
        self.x_vector_only = x_vector_only  # NEW: control ICL vs x-vector only
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.sample_rate = 16000  # Modern models use 16kHz
        self._initialized = False
        
        # Initialize TTS immediately on construction
        self._init_tts()
    
    def _init_tts(self):
        """Initialize TTS - supports Qwen3TTS (cloning) and CoquiTTS (fallback)"""
        if self._initialized:
            return
        
        try:
            from simple_tts import SimpleTTS
            
            # Map model names to SimpleTTS backends
            backend_map = {
                'simple': 'auto',  # Auto-select best available
                'qwen3tts': 'qwen3tts',
                'qwen3': 'qwen3tts',
                'coqui': 'coqui',
            }
            
            backend = backend_map.get(self.tts_model, 'auto')
            device = 'cuda:0' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'
            
            # Pass x_vector_only_mode to SimpleTTS
            self.tts = SimpleTTS(backend=backend, device=device, x_vector_only_mode=self.x_vector_only)
            self._initialized = True
            
            mode_desc = "x-vector only" if self.x_vector_only else "ICL"
            if self.tts.available_backend:
                self.logger.info(f"✓ TTS initialized: {self.tts.available_backend} (mode: {mode_desc})")
            else:
                self.logger.error("✗ TTS initialization failed - no backend available")
                
        except Exception as e:
            self.logger.error(f"TTS initialization error: {e}")
            self._initialized = False
        
    
    # ============================================================
    # SPANISH TEXT TEMPLATES
    # ============================================================
    
    SPANISH_TEXTS = {
        'professional': [
            # Forensic/formal phrases
            "Soy el locutor que necesita verificar su identidad.",
            "Mi nombre es locutor número uno.",
            "Confirmo mi identidad como locutor.",
            "Procedo a identificarme como locutor de referencia.",
            "Autorizo la verificación de mi voz.",
            "Acepto el análisis forense de mi locución.",
        ],
        'casual': [
            # Conversational
            "Hola, ¿cómo estás? Me llamo locutor.",
            "Buenos días, soy locutor número uno.",
            "Esto es mi voz, espero que funcione.",
            "Aquí estoy, lista para ser identificada.",
            "Mi voz es única y reconocible.",
            "Prueba a verificar quien soy ahora.",
        ],
        'numbers': [
            # Digit sequences (important for forensics)
            "Dígitos: uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve, cero.",
            "Números: dos, tres, cinco, siete, once, trece, diecinueve.",
            "Secuencia: cero, uno, uno, cero, uno, uno, uno, cero.",
            "Teléfono: nueve, uno, cinco, seis, dos, tres, cuatro.",
        ],
        'phonetic_diverse': [
            # Phonetically diverse (covers more linguistic space)
            "La lluvia cayó sobre la montaña.",
            "Xenia y Yaritza ejecutan tareas zonales.",
            "El queche queda quieto en el puerto.",
            "Belleza y gracia en la danza.",
            "Justicia para todos los ciudadanos.",
            "Nadie comprende el futuro incierto.",
        ],
        'read_sentences': [
            # Standard read sentences corpus
            "Barcelona es una ciudad muy bonita al lado del mar.",
            "El gato durmía tranquilamente en la sala.",
            "Mañana iremos al cine para ver una película.",
            "Los niños juegan alegremente en el parque.",
            "El café está muy caliente esta mañana.",
            "Viajaremos a Madrid el próximo mes.",
        ]
    }
    
    # ============================================================
    # SPOOFING DIFFICULTY LEVELS
    # ============================================================
    
    DIFFICULTY_LEVELS = {
        'easy': {
            'description': 'Simple, short utterances with basic text',
            'texts': ['professional'],
            'variations': 1
        },
        'medium': {
            'description': 'Mixed formal and casual utterances',
            'texts': ['professional', 'casual'],
            'variations': 2
        },
        'hard': {
            'description': 'Phonetically diverse, long utterances, with numbers',
            'texts': ['professional', 'casual', 'numbers', 'phonetic_diverse'],
            'variations': 2
        },
        'expert': {
            'description': 'Complete corpus with all sentence types',
            'texts': ['professional', 'casual', 'numbers', 'phonetic_diverse', 'read_sentences'],
            'variations': 3
        }
    }
    
    def generate_synthetic_speaker(self,
                                  speaker_name: str,
                                  speaker_wav: str,
                                  output_dir: str = './synthetic_speakers/',
                                  language: str = 'es',
                                  difficulty: str = 'medium',
                                  ref_text: Optional[str] = None) -> Dict:
        """
        Generate multiple synthetic utterances for a speaker using voice cloning
        
        Args:
            speaker_name: Name/ID of speaker to clone
            speaker_wav: Path to reference audio sample for cloning
            output_dir: Where to save synthetic audio
            language: Language code (default 'es' for Spanish)
            difficulty: 'easy', 'medium', 'hard', 'expert'
            ref_text: Optional text corresponding to speaker_wav (for better cloning).
                      If not provided, uses a generic reference text.
            
        Returns:
            Dictionary with generation metadata and file paths
        """
        if difficulty not in self.DIFFICULTY_LEVELS:
            raise ValueError(f"Difficulty must be one of {list(self.DIFFICULTY_LEVELS.keys())}")
        
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Reference wav not found: {speaker_wav}")
        
        # Create output structure
        speaker_dir = os.path.join(output_dir, speaker_name, 'synthetic')
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Default reference text if not provided
        # This is a generic text that should work for any speaker
        if ref_text is None:
            ref_text = "Hola, soy el locutor de referencia para este sistema de clonación de voz."
        
        metadata = {
            'speaker': speaker_name,
            'reference_wav': speaker_wav,
            'reference_text': ref_text,
            'difficulty': difficulty,
            'language': language,
            'generated_files': [],
            'text_sources': [],
            'generation_stats': {
                'total_texts': 0,
                'successful': 0,
                'failed': 0,
                'total_duration': 0.0
            }
        }
        
        # Get text sources for this difficulty
        difficulty_config = self.DIFFICULTY_LEVELS[difficulty]
        text_categories = difficulty_config['texts']
        
        # Collect all texts to generate
        texts_to_generate = []
        for category in text_categories:
            base_texts = self.SPANISH_TEXTS[category]
            variations = difficulty_config['variations']
            
            # Repeat texts for multiple variations
            for _ in range(variations):
                texts_to_generate.extend(base_texts)
        
        metadata['generation_stats']['total_texts'] = len(texts_to_generate)
        
        self.logger.info(f"🎙️ Generating {len(texts_to_generate)} synthetic utterances for {speaker_name}")
        self.logger.info(f"    Reference: {speaker_wav}")
        self.logger.info(f"    Difficulty: {difficulty}")
        self.logger.info(f"    Mode: {'x-vector only' if self.x_vector_only else 'ICL'}")
        
        # Initialize TTS on first use (lazy loading)
        self._init_tts()
        
        # Generate synthetic speech
        for idx, text in enumerate(texts_to_generate):
            try:
                output_file = os.path.join(
                    speaker_dir,
                    f'{speaker_name}_synthetic_{idx:03d}.wav'
                )
                
                # Generate cloned speech using speaker_wav as reference
                # This uses voice cloning to match the reference speaker's voice
                # Pass x_vector_only to control ICL vs x-vector only mode
                success = self.tts.generate_speech(
                    text=text,
                    output_file=output_file,
                    language=language,
                    ref_audio=speaker_wav,  # Reference speaker for cloning
                    ref_text=ref_text if not self.x_vector_only else None,  # Only needed for ICL mode
                    x_vector_only=self.x_vector_only  # Control mode
                )
                
                # Verify generation and get duration
                if success and os.path.exists(output_file):
                    signal, sr = librosa.load(output_file, sr=self.sample_rate)
                    duration = len(signal) / sr
                    
                    metadata['generated_files'].append({
                        'file': output_file,
                        'text': text,
                        'duration': duration
                    })
                    metadata['generation_stats']['successful'] += 1
                    metadata['generation_stats']['total_duration'] += duration
                    
                    self.logger.info(f"    ✓ [{idx+1}/{len(texts_to_generate)}] {output_file}")
                else:
                    metadata['generation_stats']['failed'] += 1
                    self.logger.warning(f"    ✗ Failed to generate: {output_file}")
                    
            except Exception as e:
                metadata['generation_stats']['failed'] += 1
                self.logger.error(f"    ERROR generating {idx}: {str(e)}")
        
        # Save metadata
        metadata_file = os.path.join(speaker_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            # Convert to JSON-serializable format
            json_metadata = metadata.copy()
            json_metadata['generation_stats']['total_duration'] = float(json_metadata['generation_stats']['total_duration'])
            json.dump(json_metadata, f, indent=2, ensure_ascii=False)
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info(f"✓ SYNTHETIC SPEAKER GENERATION COMPLETE")
        self.logger.info(f"  Speaker: {speaker_name}")
        self.logger.info(f"  Generated: {metadata['generation_stats']['successful']}/{metadata['generation_stats']['total_texts']} files")
        self.logger.info(f"  Total duration: {metadata['generation_stats']['total_duration']:.1f} seconds")
        self.logger.info(f"  Output directory: {speaker_dir}")
        self.logger.info("="*60 + "\n")
        
        return metadata
    
    def evaluate_spoofing_robustness(self,
                                    speaker_id_func,
                                    speaker_name: str,
                                    synthetic_dir: str,
                                    enrollment_file: str,
                                    threshold: float = 0.5) -> Dict:
        """
        Evaluate if speaker ID system accepts/rejects synthetic attempts
        
        Args:
            speaker_id_func: Function that identifies speaker (returns confidence 0-1)
            speaker_name: Name of target speaker
            synthetic_dir: Directory with synthetic audio files
            enrollment_file: Reference enrollment file
            threshold: Decision threshold
            
        Returns:
            Spoofing evaluation results
        """
        results = {
            'speaker': speaker_name,
            'total_attempts': 0,
            'false_accepts': 0,  # Synthetic accepted as real
            'false_accepts_rate': 0.0,  # FAR for spoofing
            'details': []
        }
        
        self.logger.info(f"\n📊 Evaluating spoofing robustness for {speaker_name}")
        self.logger.info(f"   Threshold: {threshold}")
        
        # Process all synthetic files
        synthetic_files = [
            os.path.join(synthetic_dir, f)
            for f in os.listdir(synthetic_dir)
            if f.endswith('.wav')
        ]
        
        results['total_attempts'] = len(synthetic_files)
        
        for synthetic_file in synthetic_files:
            try:
                # Get confidence score
                confidence = speaker_id_func(synthetic_file)
                
                # Check if false accept
                false_accept = confidence >= threshold
                if false_accept:
                    results['false_accepts'] += 1
                
                results['details'].append({
                    'file': os.path.basename(synthetic_file),
                    'confidence': float(confidence),
                    'accepted': false_accept
                })
                
                self.logger.info(
                    f"  {'⚠️ ACCEPT' if false_accept else '✓ REJECT'}: "
                    f"{os.path.basename(synthetic_file)[:20]:20} "
                    f"conf={confidence:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"  ERROR processing {synthetic_file}: {str(e)}")
        
        # Calculate rates
        if results['total_attempts'] > 0:
            results['false_accepts_rate'] = results['false_accepts'] / results['total_attempts']
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info(f"SPOOFING ROBUSTNESS EVALUATION")
        self.logger.info(f"  Total synthetic attempts: {results['total_attempts']}")
        self.logger.info(f"  False accepts (spoofing success): {results['false_accepts']}")
        self.logger.info(f"  Spoofing False Accept Rate (FAR): {results['false_accepts_rate']:.1%}")
        self.logger.info("="*60)
        
        if results['false_accepts_rate'] > 0.1:
            self.logger.warning("⚠️  System is vulnerable to voice cloning spoofing!")
        else:
            self.logger.info("✓ System correctly rejects synthetic spoofing attempts")
        
        return results
    
    def batch_generate_all_speakers(self,
                                   enrollment_dir: str,
                                   output_dir: str = './synthetic_speakers/',
                                   language: str = 'es',
                                   difficulty: str = 'medium') -> Dict:
        """
        Generate synthetic speakers for all enrolled speakers
        
        Args:
            enrollment_dir: Path to enrollment directory (e.g., './enrollment/')
            output_dir: Where to save synthetic audio
            language: Language code
            difficulty: Generation difficulty level
            
        Returns:
            Dictionary with all generation results
        """
        results = {
            'total_speakers': 0,
            'successful': 0,
            'failed': 0,
            'speakers': {}
        }
        
        self.logger.info(f"🎙️ Starting batch synthetic generation")
        
        # Find all speakers in enrollment directory
        if not os.path.exists(enrollment_dir):
            raise FileNotFoundError(f"Enrollment directory not found: {enrollment_dir}")
        
        for speaker_folder in os.listdir(enrollment_dir):
            speaker_path = os.path.join(enrollment_dir, speaker_folder)
            if not os.path.isdir(speaker_path):
                continue
            
            results['total_speakers'] += 1
            
            # Find first .wav file as reference
            reference_wav = None
            for filename in os.listdir(speaker_path):
                if filename.endswith('.wav'):
                    reference_wav = os.path.join(speaker_path, filename)
                    break
            
            if reference_wav is None:
                self.logger.warning(f"No .wav file found for speaker {speaker_folder}")
                results['failed'] += 1
                continue
            
            try:
                metadata = self.generate_synthetic_speaker(
                    speaker_name=speaker_folder,
                    speaker_wav=reference_wav,
                    output_dir=output_dir,
                    language=language,
                    difficulty=difficulty
                )
                results['speakers'][speaker_folder] = metadata
                results['successful'] += 1
            except Exception as e:
                self.logger.error(f"Error generating synthetic for {speaker_folder}: {e}")
                results['failed'] += 1
        
        # Save batch results
        results_file = os.path.join(output_dir, 'batch_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"\n✓ Batch generation complete: {results['successful']}/{results['total_speakers']}")
        self.logger.info(f"   Results saved to: {results_file}")
        
        return results


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def generate_spoofing_examples(speaker_name: str,
                              speaker_wav: str,
                              output_dir: str = './synthetic_speakers/') -> Dict:
    """Quick helper to generate synthetic speaker"""
    generator = TTSSpoofingGenerator()
    return generator.generate_synthetic_speaker(
        speaker_name=speaker_name,
        speaker_wav=speaker_wav,
        output_dir=output_dir,
        difficulty='medium'
    )


def evaluate_system_spoofing(speaker_id_function,
                            speaker_name: str,
                            synthetic_dir: str,
                            enrollment_file: str) -> Dict:
    """Quick helper to evaluate spoofing robustness"""
    generator = TTSSpoofingGenerator()
    return generator.evaluate_spoofing_robustness(
        speaker_id_func=speaker_id_function,
        speaker_name=speaker_name,
        synthetic_dir=synthetic_dir,
        enrollment_file=enrollment_file
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Generate synthetic speaker
    print("="*70)
    print("TTS SPOOFING EXAMPLE")
    print("="*70)
    
    generator = TTSSpoofingGenerator()
    
    # Generate synthetic versions of SPK1
    if os.path.exists('./enrollment/SPK1/SPK1_0001.wav'):
        metadata = generator.generate_synthetic_speaker(
            speaker_name='SPK1_SPOOFED',
            speaker_wav='./enrollment/SPK1/SPK1_0001.wav',
            difficulty='medium'
        )
        print("\nGeneration successful!")
        print(f"Output directory: ./synthetic_speakers/SPK1_SPOOFED/synthetic/")
    else:
        print("ERROR: Reference file ./enrollment/SPK1/SPK1_0001.wav not found")
        print("Please ensure enrollment files exist")
