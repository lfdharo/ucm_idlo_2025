import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from typing import Dict, List, Tuple
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
        threshold (float): Similarity threshold for verification decisions
        batch_size (int): Number of pairs to process in each batch
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from utils import find_files
    
    metrics = EvaluationMetrics(model_name, threshold)

    # Get all test files
    test_files = find_files(test_dir)
    
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
            # Evaluate the pair
            metrics.evaluate_pair(is_same_speaker, similarity_score)
            metrics.logger.info(f"Processing {file1}: "
                                f"Score={similarity_score:.4f}, "
                                f"Decision={'Same' if (similarity_score >= threshold and is_same_speaker == True) else 'Different'}")

            # Log progress
            if metrics.total_pairs % 100 == 0:
                metrics.logger.info(f"Processed {metrics.total_pairs}/{len(test_files)} pairs")
    
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

def show_confusion_matrix_faiss(model_name: str, test_dir: str,
                         faiss_index: object = None,
                         threshold: int = 0.5, batch_size: int = 32) -> None:
    """Show confusion matrix for a speaker verification model.
    
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
    
    # Prepare data for confusion matrix
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