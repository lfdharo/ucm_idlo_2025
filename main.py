# """
# AUTHOR: Luis F. D'Haro
# DATE: Mar 20, 2025
# PURPOSE: This script is intented to be used for the class: 
# Identificación de Locutores - Máster Lingüística y Tecnologías - UCM/UPM.
# """

from models import ModelFactory
from data_augmentation import DataAugmentation
from faiss_class import FaissClass
from evaluation import evaluate_model, plot_roc_curve, find_optimal_threshold, show_confusion_matrix
import os

model_name = 'deepspeaker'
speaker_id = 'SPK1'


# Load model
model, feature_extractor = ModelFactory.create_model(model_name)

# Initialize FAISS class
faiss_class = FaissClass(model_name=model_name, model=model, feature_extractor=feature_extractor, threshold=0.5)

# Optional: Data augmentantion
# data_augmentation = DataAugmentation(main_path='./')
# data_augmentation.augment_data(f'enrollment/{speaker_id}', 'gaussianNoise')
# data_augmentation.augment_data(f'enrollment/{speaker_id}', 'timeStretch')
# data_augmentation.augment_data(f'enrollment/{speaker_id}', 'pitchShift')
# data_augmentation.augment_data(f'enrollment/{speaker_id}', 'shift')
# data_augmentation.augment_data(f'enrollment/{speaker_id}', 'all')

# Build index from enrollment files
if os.path.exists(f'./enrollment_index_{model_name}'):
    faiss_class.load_index(f'./enrollment_index_{model_name}')
else:
    faiss_class.build_index('./enrollment')
    faiss_class.save_index(f'./enrollment_index_{model_name}')

# Verify a speaker
result = faiss_class.verify_speaker('./test/SPK1_A.wav')
# print(f"Matched speaker: {result['matched_speaker']}")
# print(f"Similarity score: {result['similarity_score']}")
# print(f"Is match: {result['is_match']}")

# Evaluate model
metrics = evaluate_model(model_name, './test', model, feature_extractor)
# print(f"Accuracy: {metrics['accuracy']}")
# print(f"Precision: {metrics['precision']}")
# print(f"Recall: {metrics['recall']}")
# print(f"F1 Score: {metrics['f1']}")

# Show confusion matrix
show_confusion_matrix(model_name, './test', model, feature_extractor)

# Plot ROC curve
plot_roc_curve(model_name, './test', model, feature_extractor)

# Find optimal threshold
find_optimal_threshold(model_name,  './test', model, feature_extractor)