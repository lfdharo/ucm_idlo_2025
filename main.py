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
from evaluation import evaluate_model_faiss, \
                        plot_roc_curve_faiss, \
                        find_optimal_threshold_faiss, \
                        show_confusion_matrix_faiss, \
                        plot_det_curve, \
                        analyze_speaker_performance, \
                        calculate_display_eer
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Speaker Verification using FAISS')
parser.add_argument('--model_name', type=str, default='wavLM', help='Model name (wavLM, SpeechBrain, Whisper)')
parser.add_argument('--speaker_id', type=str, default=None, help='Speaker ID for data augmentation')
parser.add_argument('--data_augmentation', type=str, default='gaussianNoise', help='Data augmentation type (gaussianNoise, timeStretch, pitchShift, shift, all)')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for verification')
args = parser.parse_args()
# Set parameters

# model_name = 'wavLM'  # Options: wavLM, SpeechBrain, Whisper
model_name = args.model_name
speaker_id = args.speaker_id
threshold = args.threshold

# Optional: Data augmentantion
data_augmentation = DataAugmentation(main_path='./')
# To use augmentation on specific speaker, use the speaker_id in this case on the test set
# Available augmentations: gaussianNoise, timeStretch, pitchShift, shift, all
if args.data_augmentation == 'gaussianNoise':
    data_augmentation.augment_data('test/', 'gaussianNoise', speaker_id)
elif args.data_augmentation == 'timeStretch':
    data_augmentation.augment_data('test/', 'timeStretch', speaker_id)
elif args.data_augmentation == 'pitchShift':
    data_augmentation.augment_data('test/', 'pitchShift', speaker_id)
elif args.data_augmentation == 'shift':
    data_augmentation.augment_data('test/', 'shift', speaker_id)
elif args.data_augmentation == 'all':
    data_augmentation.augment_data('test/', 'all', speaker_id)

# To use data augmentation on all speakers, use the following line
#data_augmentation.augment_data('test/', 'gaussianNoise')
# To use data augmentation on a specific speaker on the enrollment set
#data_augmentation.augment_data(f'enrollment/{speaker_id}', 'timeStretch')

# Load DNN-based model and feature extractor if needed
model, feature_extractor = ModelFactory.create_model(model_name)

# Initialize FAISS class
faiss_class = FaissClass(model_name=model_name, model=model, feature_extractor=feature_extractor, threshold=threshold)

# if the index exists, load it
# Otherwise, build the index from enrollment files
if os.path.exists(f'./enrollment_index_{model_name}'):
    faiss_class.load_index(f'./enrollment_index_{model_name}')
else: 
    faiss_class.build_index('./enrollment')
    faiss_class.save_index(f'./enrollment_index_{model_name}')

# Verify speaker with a sample audio file
result = faiss_class.verify_speaker('./test/SPK4_A.wav', threshold=threshold)
print(f"Matched speaker: {result['matched_speaker']}")
print(f"Similarity score: {result['similarity_score']}")
print(f"Is match: {result['is_match']}")

# # Evaluate model on all files in the test set
# metrics = evaluate_model_faiss(model_name, './test', faiss_class, batch_size=5, threshold=threshold)
# print(f"Accuracy: {metrics['accuracy']}")
# print(f"Precision: {metrics['precision']}")
# print(f"Recall: {metrics['recall']}")
# print(f"F1 Score: {metrics['f1']}")

# # Show confusion matrix
# show_confusion_matrix_faiss(model_name, './test', faiss_class, batch_size=5, threshold=threshold)

# # Plot ROC curve: TPR vs FPR
# plot_roc_curve_faiss(model_name, './test', faiss_class, batch_size=5)

calculate_display_eer(model_name, './test', faiss_class, batch_size=5)

# Plot DET curve
plot_det_curve(model_name, './test', faiss_class, batch_size=5)

# Analyze speaker-specific performance
speaker_metrics = analyze_speaker_performance(
    model_name=model_name,
    test_dir='./test',
    faiss_index=faiss_class,
    threshold=threshold,
    batch_size=5
)

# Find optimal threshold
find_optimal_threshold_faiss(model_name,  './test', faiss_class)
exit()
