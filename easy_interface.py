"""
Easy-to-Use Speaker Identification Interface for Students

This module provides a simplified interface for speaker identification tasks,
designed for students with limited Python experience. It handles all the 
complexity of model loading, FAISS indexing, and evaluation internally.

Author: Luis F. D'Haro
Date: Apr 7, 2026
Course: Identificación de Locutores - Máster Lingüística y Tecnologías - UCM/UPM
"""

import os
import logging
from typing import Dict, Optional, List, Tuple
import numpy as np
from pathlib import Path

from models import ModelFactory
from faiss_class import FaissClass
from evaluation import evaluate_model_faiss, analyze_speaker_performance


class SimpleSpeakerID:
    """Easy-to-use speaker identification system for students."""
    
    def __init__(self, 
                 model_name: str = 'wavLM',
                 enrollment_dir: str = './enrollment/',
                 test_dir: str = './test/',
                 threshold: float = 0.5,
                 verbose: bool = True):
        """Initialize the speaker identification system.
        
        Args:
            model_name (str): Model to use ('wavLM', 'SpeechBrain', 'Whisper')
            enrollment_dir (str): Path to enrollment audio directory
            test_dir (str): Path to test audio directory
            threshold (float): Similarity threshold (0-1). Default 0.5.
            verbose (bool): Print detailed information
            
        Example:
            >>> system = SimpleSpeakerID(model_name='wavLM')
            >>> result = system.identify('test_audio.wav')
            >>> print(result)
        """
        self.model_name = model_name
        self.enrollment_dir = enrollment_dir
        self.test_dir = test_dir
        self.threshold = threshold
        self.verbose = verbose
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
            
        # Validate directories
        if not os.path.exists(enrollment_dir):
            raise FileNotFoundError(f"Enrollment directory not found: {enrollment_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
            
        self.logger.info(f"Initializing with model: {model_name}")
        
        # Load model
        self.model, self.feature_extractor = ModelFactory.create_model(model_name)
        
        # Initialize FAISS
        self.faiss_engine = FaissClass(
            model_name=model_name,
            model=self.model,
            feature_extractor=self.feature_extractor,
            threshold=threshold
        )
        
        # Build or load index
        self._setup_faiss_index()
        self.logger.info("✓ System ready for speaker identification")
    
    def _setup_faiss_index(self):
        """Load existing FAISS index or build a new one."""
        index_path = f'{self.enrollment_dir}/enrollment_index_{self.model_name}'
        
        if os.path.exists(index_path):
            self.logger.info("Loading existing FAISS index...")
            self.faiss_engine.load_index(index_path)
            self.logger.info("✓ Index loaded")
        else:
            self.logger.info("Building FAISS index from enrollment files...")
            self.faiss_engine.build_index(self.enrollment_dir)
            self.faiss_engine.save_index(index_path)
            self.logger.info("✓ Index built and saved")
    
    def identify(self, audio_file: str) -> Dict[str, any]:
        """Identify speaker from a single audio file.
        
        Args:
            audio_file (str): Path to audio file (WAV format recommended)
            
        Returns:
            dict: Results including:
                - 'speaker_id': Matched speaker ID (or 'Unknown')
                - 'confidence': Similarity score (0-1)
                - 'is_match': Boolean (True if above threshold)
                - 'threshold_used': Threshold used for decision
                
        Example:
            >>> result = system.identify('test_audio.wav')
            >>> print(f"Speaker: {result['speaker_id']}")
            >>> print(f"Confidence: {result['confidence']:.2%}")
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        result = self.faiss_engine.verify_speaker(audio_file, self.threshold)
        
        # Format result for students
        return {
            'speaker_id': result['matched_speaker'],
            'confidence': result['similarity_score'],
            'is_match': result['is_match'],
            'threshold_used': self.threshold,
            'message': self._format_result_message(result)
        }
    
    def identify_batch(self, audio_files: List[str]) -> List[Dict]:
        """Identify speakers for multiple audio files.
        
        Args:
            audio_files (list): List of paths to audio files
            
        Returns:
            list: List of result dictionaries (same format as identify())
            
        Example:
            >>> files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
            >>> results = system.identify_batch(files)
            >>> for r in results:
            ...     print(f"{r['file']}: {r['speaker_id']}")
        """
        results = []
        self.logger.info(f"Processing {len(audio_files)} files...")
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                result = self.identify(audio_file)
                result['file'] = audio_file
                result['file_number'] = i
                results.append(result)
                self.logger.info(f"[{i}/{len(audio_files)}] {audio_file}: {result['speaker_id']}")
            except Exception as e:
                self.logger.error(f"Error processing {audio_file}: {e}")
                results.append({
                    'file': audio_file,
                    'error': str(e),
                    'speaker_id': 'Error'
                })
        
        return results
    
    def evaluate(self) -> Dict[str, any]:
        """Evaluate system performance on all test files.
        
        Returns:
            dict: Performance metrics including:
                - 'accuracy': Overall accuracy (0-1)
                - 'precision': Precision score (0-1)
                - 'recall': Recall score (0-1)
                - 'f1_score': F1 score (0-1)
                - 'total_files': Total test files processed
                - 'by_speaker': Performance per speaker
                
        Example:
            >>> metrics = system.evaluate()
            >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
            >>> for spk, perf in metrics['by_speaker'].items():
            ...     print(f"{spk}: {perf['accuracy']:.2%}")
        """
        self.logger.info(f"Evaluating on test set ({self.test_dir})...")
        
        # Get overall metrics
        metrics = evaluate_model_faiss(
            self.model_name,
            self.test_dir,
            self.faiss_engine,
            batch_size=5,
            threshold=self.threshold
        )
        
        # Get per-speaker metrics
        speaker_metrics = analyze_speaker_performance(
            model_name=self.model_name,
            test_dir=self.test_dir,
            faiss_index=self.faiss_engine,
            threshold=self.threshold,
            batch_size=5
        )
        
        results = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
            'total_files': metrics.get('total_pairs', 0),
            'by_speaker': speaker_metrics,
            'threshold': self.threshold,
            'model': self.model_name
        }
        
        self._print_evaluation_summary(results)
        return results
    
    def set_threshold(self, threshold: float):
        """Change the decision threshold.
        
        Args:
            threshold (float): New threshold (0-1)
            
        Example:
            >>> system.set_threshold(0.6)
            >>> result = system.identify('audio.wav')
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        self.faiss_engine.threshold = threshold
        self.logger.info(f"Threshold changed to {threshold}")
    
    def list_enrolled_speakers(self) -> List[str]:
        """List all enrolled speakers in the database.
        
        Returns:
            list: Speaker IDs
            
        Example:
            >>> speakers = system.list_enrolled_speakers()
            >>> print(f"Enrolled speakers: {speakers}")
        """
        speakers = set()
        for item in self.faiss_engine.speaker_ids:
            # Extract speaker ID from filename (e.g., 'SPK1_0001' -> 'SPK1')
            speaker = item.split('_')[0]
            speakers.add(speaker)
        
        return sorted(list(speakers))
    
    def calculate_eer(self) -> Tuple[float, float]:
        """Calculate Equal Error Rate (EER) for the system.
        
        EER is the point where False Acceptance Rate (FAR) equals 
        False Rejection Rate (FRR). Lower EER = better performance.
        
        Returns:
            tuple: (eer_threshold, eer_value)
                - eer_threshold: Threshold at which EER occurs
                - eer_value: The EER value (0-1)
        
        Example:
            >>> threshold, eer = system.calculate_eer()
            >>> print(f"EER: {eer:.4f}")
            >>> print(f"Threshold: {threshold:.4f}")
        """
        from evaluation import calculate_eer
        from utils import find_files
        
        self.logger.info("Calculating EER...")
        test_files = find_files(self.test_dir)
        
        y_true = []
        y_scores = []
        
        for file_path in test_files:
            spk1 = os.path.basename(file_path).split('_')[0]
            result = self.faiss_engine.verify_speaker(file_path)
            matched_speaker = result['matched_speaker'].split('_')[0]
            
            y_true.append(1 if spk1 == matched_speaker else 0)
            y_scores.append(result['similarity_score'])
        
        eer_threshold, eer_value = calculate_eer(y_true, y_scores)
        
        self.logger.info(f"✓ EER: {eer_value:.4f} at threshold {eer_threshold:.4f}")
        return eer_threshold, eer_value
    
    def plot_roc_curve(self, save_to: Optional[str] = None) -> None:
        """Plot ROC curve (Receiver Operating Characteristic).
        
        Shows the trade-off between true positive rate and false positive rate
        at different thresholds. Closer to top-left corner = better performance.
        
        Args:
            save_to (str, optional): Path to save the figure
        
        Example:
            >>> system.plot_roc_curve(save_to='results/roc.png')
        """
        from evaluation import plot_roc_curve_faiss
        
        self.logger.info("Plotting ROC curve...")
        plot_roc_curve_faiss(self.model_name, self.test_dir, self.faiss_engine)
        if save_to:
            import matplotlib.pyplot as plt
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ ROC curve saved to {save_to}")
    
    def plot_det_curve(self, save_to: Optional[str] = None) -> None:
        """Plot DET curve (Detection Error Tradeoff).
        
        Similar to ROC but uses logarithmic scales and plots FAR vs FRR.
        Useful for forensic analysis applications where both false positives
        and false negatives are important.
        
        Args:
            save_to (str, optional): Path to save the figure
        
        Example:
            >>> system.plot_det_curve(save_to='results/det.png')
        """
        from evaluation import plot_det_curve
        
        self.logger.info("Plotting DET curve...")
        plot_det_curve(self.model_name, self.test_dir, self.faiss_engine)
        if save_to:
            import matplotlib.pyplot as plt
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ DET curve saved to {save_to}")
    
    def get_eer_metrics(self) -> Dict[str, float]:
        """Get system metrics at the EER point.
        
        Returns:
            dict: Metrics at EER threshold including:
                - 'eer_threshold': Threshold at EER
                - 'eer_value': The EER value
                - 'far': False Acceptance Rate
                - 'frr': False Rejection Rate
        
        Example:
            >>> metrics = system.get_eer_metrics()
            >>> print(f"EER threshold: {metrics['eer_threshold']:.4f}")
        """
        from evaluation import calculate_eer
        from utils import find_files
        
        test_files = find_files(self.test_dir)
        
        y_true = []
        y_scores = []
        
        for file_path in test_files:
            spk1 = os.path.basename(file_path).split('_')[0]
            result = self.faiss_engine.verify_speaker(file_path)
            matched_speaker = result['matched_speaker'].split('_')[0]
            
            y_true.append(1 if spk1 == matched_speaker else 0)
            y_scores.append(result['similarity_score'])
        
        eer_threshold, eer_value = calculate_eer(y_true, y_scores)
        
        # Calculate FAR and FRR at this threshold
        import numpy as np
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = (y_scores >= eer_threshold).astype(int)
        
        fp = np.sum((1 - y_true) * y_pred)
        fn = np.sum(y_true * (1 - y_pred))
        tn = np.sum((1 - y_true) * (1 - y_pred))
        tp = np.sum(y_true * y_pred)
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'eer_threshold': eer_threshold,
            'eer_value': eer_value,
            'far': far,
            'frr': frr
        }
    
    def print_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("SPEAKER IDENTIFICATION SYSTEM - CONFIGURATION")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Enrollment directory: {self.enrollment_dir}")
        print(f"Test directory: {self.test_dir}")
        print(f"Threshold: {self.threshold}")
        print(f"Enrolled speakers: {', '.join(self.list_enrolled_speakers())}")
        print("="*60 + "\n")
    
    # ======================== Private Methods ========================
    
    def _format_result_message(self, result: Dict) -> str:
        """Format result into a human-readable message."""
        if result['is_match']:
            return f"✓ MATCH: Speaker identified as {result['matched_speaker']} with {result['similarity_score']:.1%} confidence"
        else:
            return f"✗ NO MATCH: Highest similarity is {result['matched_speaker']} with {result['similarity_score']:.1%} confidence (below {self.threshold} threshold)"
    
    def _print_evaluation_summary(self, results: Dict):
        """Print a nice summary of evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {results['model']}")
        print(f"Threshold: {results['threshold']}")
        print(f"Total test files: {results['total_files']}")
        print()
        print("OVERALL PERFORMANCE:")
        print(f"  Accuracy:  {results['accuracy']:.2%}")
        print(f"  Precision: {results['precision']:.2%}")
        print(f"  Recall:    {results['recall']:.2%}")
        print(f"  F1 Score:  {results['f1_score']:.2%}")
        
        if results['by_speaker']:
            print("\nPER-SPEAKER PERFORMANCE:")
            for speaker, metrics in results['by_speaker'].items():
                acc = metrics.get('accuracy', 0)
                print(f"  {speaker}: {acc:.2%}")
        
        print("="*60 + "\n")
