"""
Speaker Verification Evaluation Module

This module provides tools for evaluating speaker verification systems using various metrics:
- Accuracy, Precision, Recall, and F1 Score
- ROC Curve Analysis
- Confusion Matrix Visualization
- Optimal Threshold Finding

The module supports both traditional model evaluation and FAISS-based evaluation.

Author: Luis F. D'Haro
Date: Mar 20, 2025
Course: Identificación de Locutores - Máster Lingüística y Tecnologías - UCM/UPM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from typing import Dict, List, Tuple, Optional, Union
import logging

class EvaluationMetrics:
    def __init__(self, model_name: str, threshold: float = 0.6):
        """Initialize evaluation metrics class.
        
        Args:
            model_name (str): Name of the model being evaluated
            threshold (float): Similarity threshold for verification decisions
        """
        self.model_name = model_name
        self.threshold = threshold
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        self.total_pairs = 0
        self.logger = logging.getLogger(__name__)       

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate verification metrics based on current results.
        
        Returns:
            dict: Dictionary containing accuracy, precision, recall, and F1 score
        """
        accuracy = (self.tp + self.tn) / self.total_pairs if self.total_pairs > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.logger.info(f"Metrics for {self.model_name} (threshold={self.threshold}):")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
            
        return metrics
    
    def evaluate_pair(self, is_same_speaker: bool, similarity_score: float) -> None:
        """Evaluate a single speaker verification pair.
        
        Args:
            is_same_speaker (bool): True if the pair is from same speaker
            similarity_score (float): Similarity score between the pair
        """
        self.total_pairs += 1
        decision = similarity_score >= self.threshold
        
        if is_same_speaker:
            if decision:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if decision:
                self.fp += 1
            else:
                self.tn += 1

def evaluate_model_faiss(model_name: str, test_dir: str,
                   faiss_index: object = None,
                   threshold: float = 0.6, batch_size: int = 32) -> Dict[str, float]:
    """Evaluate a speaker verification model using FAISS.
    
    Args:
        model_name (str): Name of the model to evaluate
        test_dir (str): Directory containing test audio files
        faiss_index (object): FAISS index object
        threshold (float): Similarity threshold for verification decisions (0-1)
        batch_size (int): Number of pairs to process in each batch
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from utils import find_files
    
    metrics = EvaluationMetrics(model_name, threshold)

    # Get all test files
    test_files = find_files(test_dir)
    if not test_files:
        raise ValueError(f"No audio files found in {test_dir}")
    
    # Process pairs in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        for file1 in batch_files:
            try:
                # Extract speaker IDs from filenames
                spk1 = os.path.basename(file1).split('_')[0]
              
                result = faiss_index.verify_speaker(file1)
                similarity_score = result['similarity_score']
                matched_speaker = result['matched_speaker'].split('_')[0]
                is_same_speaker = spk1 == matched_speaker            
                
                # Evaluate the pair
                metrics.evaluate_pair(is_same_speaker, similarity_score)
                metrics.logger.info(f"Processing {file1}: "
                                    f"Score={similarity_score:.4f}, "
                                    f"Decision={'Same' if (similarity_score >= threshold and is_same_speaker == True) else 'Different'}")

                # Log progress
                if metrics.total_pairs % 100 == 0:
                    metrics.logger.info(f"Processed {metrics.total_pairs}/{len(test_files)} pairs")
            except Exception as e:
                metrics.logger.error(f"Error processing file {file1}: {str(e)}")
                continue
    
    return metrics.calculate_metrics()

def plot_roc_curve_faiss(model_name: str, test_dir: str,
                   faiss_index: object = None, 
                   batch_size: int = 32) -> None:
    """Plot ROC curve for a speaker verification model.
    
    Args:
        model_name (str): Name of the model to evaluate
        test_dir (str): Directory containing test audio files
        batch_size (int): Number of pairs to process in each batch
    
    """
    from utils import find_files
  
   
    # Get all test files
    test_files = find_files(test_dir)
    
    # Prepare data for ROC curve
    y_true = []
    y_scores = []
    
    # Process pairs in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        for file1 in batch_files:
            # Extract speaker IDs from filenames
            spk1 = os.path.basename(file1).split('_')[0]
          
            result = faiss_index.verify_speaker(file1)
            similarity_score = result['similarity_score']
            matched_speaker = result['matched_speaker'].split('_')[0]
            is_same_speaker = spk1 == matched_speaker   
              
            y_true.append(1 if is_same_speaker else 0)
            y_scores.append(similarity_score)
            
            # Log progress
            if len(y_true) % 100 == 0:
                logging.info(f"Processed {len(y_true)}/{len(test_files)} pairs")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def find_optimal_threshold_faiss(model_name: str, test_dir: str, 
                           faiss_index: object = None,
                           num_thresholds: int = 100, batch_size: int = 32) -> Tuple[float, Dict[str, float]]:
    """Find the optimal threshold that maximizes F1 score.
    
    Args:
        model_name (str): Name of the model to evaluate
        test_dir (str): Directory containing test audio files
        faiss_index (object): FAISS index object
        num_thresholds (int): Number of threshold points to evaluate
        batch_size (int): Number of pairs to process in each batch
        
    Returns:
        tuple: (optimal_threshold, best_metrics)
    """
    from models import ModelFactory
    from vector_embedding import exctract_vector_embedding
    from utils import find_files
        
    # Get all test files
    test_files = find_files(test_dir)
   
    # Prepare data for threshold search
    y_true = []
    y_scores = []
    
    # Process pairs in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        for file1 in batch_files:
            # Extract speaker IDs from filenames
            spk1 = os.path.basename(file1).split('_')[0]
          
            result = faiss_index.verify_speaker(file1)
            similarity_score = result['similarity_score']
            matched_speaker = result['matched_speaker'].split('_')[0]
            is_same_speaker = spk1 == matched_speaker   
              
            y_true.append(1 if is_same_speaker else 0)
            y_scores.append(similarity_score)
            
            # Log progress
            if len(y_true) % 100 == 0:
                logging.info(f"Processed {len(y_true)}/{len(test_files)} pairs")           
   
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for threshold in np.linspace(min(y_scores), max(y_scores), num_thresholds):
        metrics = EvaluationMetrics(model_name, threshold)
        for score, true_label in zip(y_scores, y_true):
            metrics.evaluate_pair(true_label == 1, score)
        
        current_metrics = metrics.calculate_metrics()
        if current_metrics['f1'] > best_f1:
            best_f1 = current_metrics['f1']
            best_threshold = threshold
            best_metrics = current_metrics
    
    logging.info(f"Optimal threshold for {model_name}: {best_threshold:.4f}")
    logging.info(f"Best metrics at optimal threshold:")
    for metric, value in best_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return best_threshold, best_metrics 

# def show_confusion_matrix_faiss(model_name: str, test_dir: str,
#                          faiss_index: object = None,
#                          threshold: int = 0.5, batch_size: int = 32) -> None:
#     """Show confusion matrix for a speaker verification model.
    
#     Args:
#         model_name (str): Name of the model to evaluate
#         test_dir (str): Directory containing test audio files
#         faiss_index (object): FAISS index object
#         threshold (float): Similarity threshold for verification decisions
#         batch_size (int): Number of pairs to process in each batch
#     """
#     from sklearn.metrics import confusion_matrix
#     from utils import find_files
#     import seaborn as sns 
    
#     # Get all test files
#     test_files = find_files(test_dir)
    
#     # Prepare data for confusion matrix
#     y_true = []
#     y_scores = []
    
#     # Process pairs in batches
#     for i in range(0, len(test_files), batch_size):
#         batch_files = test_files[i:i+batch_size]
#         for file1 in batch_files:
#             # Extract speaker IDs from filenames
#             spk1 = os.path.basename(file1).split('_')[0]
          
#             result = faiss_index.verify_speaker(file1, threshold=threshold)
#             similarity_score = result['similarity_score']
#             is_same_speaker = result['is_match']   
              
#             y_true.append(1 if is_same_speaker else 0)
#             y_scores.append(similarity_score)
            
#             # Log progress
#             if len(y_true) % 100 == 0:
#                 logging.info(f"Processed {len(y_true)}/{len(test_files)} pairs")
    
#     # Create confusion matrix
#     cm = confusion_matrix(y_true, [1 if score >= threshold else 0 for score in y_scores])
    
#     # Plot confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
#                 xticklabels=['Not Same Speaker', 'Same Speaker'],
#                 yticklabels=['Not Same Speaker', 'Same Speaker'])
#     plt.title(f'Confusion Matrix - {model_name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()
#     plt.close()


def show_confusion_matrix_faiss(model_name: str, test_dir: str,
                         faiss_index: object = None,
                         threshold: float = 0.5, batch_size: int = 32) -> None:
    """Show confusion matrix for a speaker verification model and log file classifications.
    
    Args:
        model_name (str): Name of the model to evaluate
        test_dir (str): Directory containing test audio files
        faiss_index (object): FAISS index object
        threshold (float): Similarity threshold for verification decisions
        batch_size (int): Number of pairs to process in each batch
    """
    from sklearn.metrics import confusion_matrix
    from utils import find_files
    import seaborn as sns 
    
    # Get all test files
    test_files = find_files(test_dir)
    
    # Prepare data for confusion matrix and file tracking
    y_true = []
    y_scores = []
    file_classifications = {"TP": [], "FP": [], "FN": [], "TN": []}
    
    # Process pairs in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        for file1 in batch_files:
            # Extract speaker IDs from filenames
            spk1 = os.path.basename(file1).split('_')[0]
          
            result = faiss_index.verify_speaker(file1, threshold=threshold)
            similarity_score = result['similarity_score']
            is_same_speaker = result['is_match']   
              
            y_true.append(1 if is_same_speaker else 0)
            y_scores.append(similarity_score)
            
            # Classify the file based on the threshold
            predicted = 1 if similarity_score >= threshold else 0
            if is_same_speaker and predicted == 1:
                file_classifications["TP"].append(file1)
            elif is_same_speaker and predicted == 0:
                file_classifications["FN"].append(file1)
            elif not is_same_speaker and predicted == 1:
                file_classifications["FP"].append(file1)
            elif not is_same_speaker and predicted == 0:
                file_classifications["TN"].append(file1)
            
            # Log progress
            if len(y_true) % 100 == 0:
                logging.info(f"Processed {len(y_true)}/{len(test_files)} pairs")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, [1 if score >= threshold else 0 for score in y_scores])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=['Not Same Speaker', 'Same Speaker'],
                yticklabels=['Not Same Speaker', 'Same Speaker'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.close()
    
    # Log file classifications
    logging.info("File classifications:")
    for classification, files in file_classifications.items():
        logging.info(f"{classification} ({len(files)} files):")
        for file in files:
            logging.info(f"  {file}")


def calculate_eer(y_true: List[int], y_scores: List[float]) -> Tuple[float, float]:
    """Calculate Equal Error Rate (EER) for speaker verification.
    
    The EER is the point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).
    
    Args:
        y_true (List[int]): List of true labels (1 for same speaker, 0 for different)
        y_scores (List[float]): List of similarity scores
        
    Returns:
        tuple: (eer_threshold, eer_value)
    """
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Get all possible thresholds
    thresholds = np.sort(np.unique(y_scores))
    
    # Calculate FAR and FRR for each threshold
    far = []
    frr = []
    
    for threshold in thresholds:
        # Calculate predictions
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate FAR and FRR
        fp = np.sum((1 - y_true) * y_pred)  # False positives
        fn = np.sum(y_true * (1 - y_pred))  # False negatives
        tn = np.sum((1 - y_true) * (1 - y_pred))  # True negatives
        tp = np.sum(y_true * y_pred)  # True positives
        
        far_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        far.append(far_rate)
        frr.append(frr_rate)
    
    # Find the point where FAR and FRR are closest
    diff = np.abs(np.array(far) - np.array(frr))
    eer_idx = np.argmin(diff)
    
    return thresholds[eer_idx], (far[eer_idx] + frr[eer_idx]) / 2


def plot_det_curve(model_name: str, test_dir: str,
                  faiss_index: object = None,
                  batch_size: int = 32) -> None:
    """Plot Detection Error Tradeoff (DET) curve for speaker verification.
    
    DET curves are similar to ROC curves but use logarithmic scales and show
    False Acceptance Rate (FAR) vs False Rejection Rate (FRR).
    
    Args:
        model_name (str): Name of the model to evaluate
        test_dir (str): Directory containing test audio files
        faiss_index (object): FAISS index object
        batch_size (int): Number of pairs to process in each batch
    """
    from utils import find_files
    
    # Get all test files
    test_files = find_files(test_dir)
    
    # Prepare data for DET curve
    y_true = []
    y_scores = []
    
    # Process pairs in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        for file1 in batch_files:
            # Extract speaker IDs from filenames
            spk1 = os.path.basename(file1).split('_')[0]
            
            result = faiss_index.verify_speaker(file1)
            similarity_score = result['similarity_score']
            matched_speaker = result['matched_speaker'].split('_')[0]
            is_same_speaker = spk1 == matched_speaker
            
            y_true.append(1 if is_same_speaker else 0)
            y_scores.append(similarity_score)
    
    # Calculate FAR and FRR for different thresholds
    thresholds = np.sort(np.unique(y_scores))
    far = []
    frr = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        fp = sum((1 - np.array(y_true)) * np.array(y_pred))
        fn = sum(np.array(y_true) * (1 - np.array(y_pred)))
        tn = sum((1 - np.array(y_true)) * (1 - np.array(y_pred)))
        tp = sum(np.array(y_true) * np.array(y_pred))
        
        far_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        far.append(far_rate)
        frr.append(frr_rate)
    
    # Plot DET curve
    plt.figure(figsize=(8, 6))
    plt.plot(far, frr, 'b-', label=f'{model_name}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title(f'DET Curve - {model_name}')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.show()


def analyze_speaker_performance(model_name: str, test_dir: str,
                              faiss_index: object = None,
                              threshold: float = 0.6,
                              batch_size: int = 32) -> Dict[str, Dict[str, float]]:
    """Analyze performance metrics for each individual speaker.
    
    Args:
        model_name (str): Name of the model to evaluate
        test_dir (str): Directory containing test audio files
        faiss_index (object): FAISS index object
        threshold (float): Similarity threshold for verification decisions
        batch_size (int): Number of pairs to process in each batch
        
    Returns:
        dict: Dictionary containing performance metrics for each speaker
    """
    from utils import find_files
    
    # Get all test files
    test_files = find_files(test_dir)
    
    # Initialize speaker-specific metrics
    speaker_metrics = {}
    
    # Process pairs in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        for file1 in batch_files:
            # Extract speaker ID
            spk1 = os.path.basename(file1).split('_')[0]
            
            # Initialize metrics for new speaker
            if spk1 not in speaker_metrics:
                speaker_metrics[spk1] = {
                    'total_samples': 0,
                    'correct_verifications': 0,
                    'false_rejections': 0,
                    'false_acceptances': 0,
                    'avg_similarity': 0.0
                }
            
            result = faiss_index.verify_speaker(file1, threshold=threshold)
            similarity_score = result['similarity_score']
            matched_speaker = result['matched_speaker'].split('_')[0]
            is_same_speaker = spk1 == matched_speaker
            
            # Update speaker metrics
            speaker_metrics[spk1]['total_samples'] += 1
            speaker_metrics[spk1]['avg_similarity'] += similarity_score
            
            if is_same_speaker:
                if similarity_score >= threshold:
                    speaker_metrics[spk1]['correct_verifications'] += 1
                else:
                    speaker_metrics[spk1]['false_rejections'] += 1
            else:
                if similarity_score >= threshold:
                    speaker_metrics[spk1]['false_acceptances'] += 1
    
    # Calculate final metrics for each speaker
    for speaker in speaker_metrics:
        metrics = speaker_metrics[speaker]
        metrics['avg_similarity'] /= metrics['total_samples']
        metrics['verification_rate'] = metrics['correct_verifications'] / metrics['total_samples']
        metrics['false_rejection_rate'] = metrics['false_rejections'] / metrics['total_samples']
        metrics['false_acceptance_rate'] = metrics['false_acceptances'] / metrics['total_samples']
    
    # Log speaker-specific results
    logging.info("\nSpeaker-specific Performance Analysis:")
    for speaker, metrics in speaker_metrics.items():
        logging.info(f"\nSpeaker {speaker}:")
        logging.info(f"  Total Samples: {metrics['total_samples']}")
        logging.info(f"  Verification Rate: {metrics['verification_rate']:.4f}")
        logging.info(f"  False Rejection Rate: {metrics['false_rejection_rate']:.4f}")
        logging.info(f"  False Acceptance Rate: {metrics['false_acceptance_rate']:.4f}")
        logging.info(f"  Average Similarity Score: {metrics['avg_similarity']:.4f}")
    
    return speaker_metrics


def calculate_display_eer(model_name: str, test_dir: str,
                         faiss_index: object = None,
                         batch_size: int = 32) -> None:
    """Calculate and display the Equal Error Rate (EER) for speaker verification.
    
    Args:
        model_name (str): Name of the model to evaluate
        test_dir (str): Directory containing test audio files
        faiss_index (object): FAISS index object
        batch_size (int): Number of pairs to process in each batch
    """
    from utils import find_files
    
    # Get all test files
    test_files = find_files(test_dir)
    
    # Prepare data for EER calculation
    y_true = []
    y_scores = []
    
    # Process pairs in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        for file1 in batch_files:
            # Extract speaker IDs from filenames
            spk1 = os.path.basename(file1).split('_')[0]
            
            result = faiss_index.verify_speaker(file1)
            similarity_score = result['similarity_score']
            matched_speaker = result['matched_speaker'].split('_')[0]
            is_same_speaker = spk1 == matched_speaker
            
            y_true.append(1 if is_same_speaker else 0)
            y_scores.append(similarity_score)
    
    # Calculate and display EER
    eer_threshold, eer_value = calculate_eer(y_true, y_scores)
    print(f"\nEqual Error Rate (EER): {eer_value:.4f}")
    print(f"EER Threshold: {eer_threshold:.4f}")
    
    return eer_threshold, eer_value