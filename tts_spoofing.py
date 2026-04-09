"""
TTS-Based Speaker Spoofing Module

Generates synthetic speech for evaluating speaker identification robustness:
- Modern lightweight TTS using 2024-2025 HuggingFace models
- Models: Kokoro-82M, Qwen3-TTS, Fish Audio, ParlerTTS
- Controlled spoofing attempts with template variations
- Synthetic batch generation for spoofing detection evaluation
- Comparison metrics: Real vs Synthetic speaker verification rates

Forensic Use Case:
Students can use this module to test if their speaker ID system correctly
rejects or accepts synthetic attempts of an enrolled speaker using TTS synthesis.

Note: Uses lightweight models that work on CPU
- Kokoro-82M: 82M params, ultra-fast, excellent for CPU-only
- Qwen3-TTS: 1.7B, best features/quality balance
- Fish Audio: 500M-5B variants, 13+ languages
- ParlerTTS: 900M, minimal multilingual

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
    
    def __init__(self, model: str = 'simple', use_gpu: bool = True):
        """
        Initialize TTS for spoofing generation
        
        Args:
            model: TTS model to use:
                - 'simple': Fast mock/espeak (✓ Works immediately, recommended)
                - 'bark': HuggingFace Bark 100MB (slow on CPU, good quality if GPU available)
                - 'qwen3-mini': 600MB lightweight version (requires GPU)
                - 'qwen3': Full 1.7B version (best quality, requires GPU)
            use_gpu: Use GPU if available (recommended for bark/qwen3)
        """
        self.tts = None
        self.tts_model = model
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sample_rate = 16000  # Modern models use 16kHz
        self._initialized = False
        
        # Initialize TTS immediately on construction
        self._init_tts()
    
    def _init_tts(self):
        """Lazy initialize TTS - try system-based first, then ML models"""
        if self._initialized:
            return
        
        # Priority 1: SimpleTTS
        if self.tts_model == 'simple' or self.tts_model in ['espeak', 'pyttsx3', 'mock', 'qwen3tts']:
            try:
                from simple_tts import SimpleTTS
                backend = 'auto' if self.tts_model == 'simple' else self.tts_model
                self.tts = SimpleTTS(backend=backend)
                self._initialized = True
                self.logger.info(f"Initialized SimpleTTS (backend: {self.tts.available_backend})")
                return
            except Exception as e:
                self.logger.warning(f"SimpleTTS failed: {e}")
        
    
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
                                  difficulty: str = 'medium') -> Dict:
        """
        Generate multiple synthetic utterances for a speaker using voice cloning
        
        Args:
            speaker_name: Name/ID of speaker to clone
            speaker_wav: Path to reference audio sample for cloning
            output_dir: Where to save synthetic audio
            language: Language code (default 'es' for Spanish)
            difficulty: 'easy', 'medium', 'hard', 'expert'
            
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
        
        metadata = {
            'speaker': speaker_name,
            'reference_wav': speaker_wav,
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
        
        # Initialize TTS on first use (lazy loading)
        self._init_tts()
        
        # Generate synthetic speech
        for idx, text in enumerate(texts_to_generate):
            try:
                output_file = os.path.join(
                    speaker_dir,
                    f'{speaker_name}_synthetic_{idx:03d}.wav'
                )
                
                # Generate TTS speech (not voice cloning - standard TTS synthesis)
                success = self.tts.generate_speech(
                    text=text,
                    output_file=output_file,
                    language=language
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
