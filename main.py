# """
# AUTHOR: Luis F. D'Haro
# DATE: Mar 20, 2025
# PURPOSE: This script is intented to be used for the class: 
# Identificación de Locutores - Máster Lingüística y Tecnologías - UCM/UPM.
# """

# import torch
from models import ModelFactory
from data_augmentation import DataAugmentation
from faiss_class import FaissClass
from evaluation import evaluate_model_faiss, plot_roc_curve_faiss, find_optimal_threshold_faiss, show_confusion_matrix_faiss
import os

# model_name = 'wavLM'  # Options: wavLM, SpeechBrain, whisper
model_name = 'wavLM'
speaker_id = 'SPK1'


# Optional: Data augmentantion
data_augmentation = DataAugmentation(main_path='./')
# To use augmentation on specific speaker, use the speaker_id in this case on the test set
# Available augmentations: gaussianNoise, timeStretch, pitchShift, shift, all
data_augmentation.augment_data('test/', 'gaussianNoise', speaker_id) 
# To use data augmentation on all speakers, use the following line
#data_augmentation.augment_data('test/', 'gaussianNoise')
# To use data augmentation on a specific speaker on the enrollment set
#data_augmentation.augment_data(f'enrollment/{speaker_id}', 'timeStretch')


# Load DNN-based model and feature extractor if needed
model, feature_extractor = ModelFactory.create_model(model_name)

# if torch.cuda.is_available():
#     model.to('cuda')
# else:
#     model.to('cpu')

# Initialize FAISS class
faiss_class = FaissClass(model_name=model_name, model=model, feature_extractor=feature_extractor, threshold=0.5)

# if the index exists, load it
# Otherwise, build the index from enrollment files
if os.path.exists(f'./enrollment_index_{model_name}'):
    faiss_class.load_index(f'./enrollment_index_{model_name}')
else: 
    faiss_class.build_index('./enrollment')
    faiss_class.save_index(f'./enrollment_index_{model_name}')

# Verify speaker with a sample audio file
result = faiss_class.verify_speaker('./test/SPK1_A.wav')
print(f"Matched speaker: {result['matched_speaker']}")
print(f"Similarity score: {result['similarity_score']}")
print(f"Is match: {result['is_match']}")

# Evaluate model on all files in the test set
metrics = evaluate_model_faiss(model_name, './test', faiss_class, batch_size=5)
print(f"Accuracy: {metrics['accuracy']}")
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1 Score: {metrics['f1']}")

# Show confusion matrix
show_confusion_matrix_faiss(model_name, './test', faiss_class, batch_size=5)

# Plot ROC curve: TPR vs FPR
plot_roc_curve_faiss(model_name, './test', faiss_class, batch_size=5)

# Find optimal threshold
find_optimal_threshold_faiss(model_name,  './test', faiss_class)
