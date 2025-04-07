# Speaker Verification System

This project implements a speaker verification system using various deep learning models and FAISS for similarity search. The system can be used to verify if a given audio sample matches the voice of a previously enrolled speaker.

## Features

- Support for multiple speaker verification models:
  - WavLM
  - SpeechBrain
  - Whisper
- FAISS-based similarity search for efficient speaker matching
- Comprehensive evaluation metrics (accuracy, precision, recall, F1 score)
- ROC curve visualization
- Automatic threshold optimization
- Batch processing support for multiple audio files

## Installation

1. Download the files from Moodle

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv_sre
source venv_sre/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `evaluation.py`: Contains evaluation metrics and visualization functions
- `faiss_class.py`: Implements FAISS-based similarity search
- `models.py`: Model factory and model-specific implementations
- `utils.py`: Utility functions for file handling and logging
- `vector_embedding.py`: Functions for extracting speaker embeddings

## Usage

1. Prepare your audio files:
   - Enrollment files should be named as `speaker_id_*.wav`
   - Test files can have any name

2. Basic usage example:
```python
from models import ModelFactory
from faiss_class import FaissClass
from evaluation import evaluate_model, plot_roc_curve

# Create model
model, feature_extractor = ModelFactory.create_model('wavLM')

# Initialize FAISS class
faiss_class = FaissClass(model_name='wavLM', threshold=0.5)

# Build index from enrollment files
faiss_class.build_index('path/to/enrollment/files')

# Verify a speaker
result = faiss_class.verify_speaker('path/to/test/file.wav')
print(f"Matched speaker: {result['matched_speaker']}")
print(f"Similarity score: {result['similarity_score']}")
print(f"Is match: {result['is_match']}")

# Evaluate model
metrics = evaluate_model('wavLM', 'path/to/enrollment/files', 'path/to/test/files')
print(f"Accuracy: {metrics['accuracy']}")
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1 Score: {metrics['f1_score']}")

# Plot ROC curve
plot_roc_curve('wavLM', 'path/to/enrollment/files', 'path/to/test/files')
```

## Evaluation Metrics

The system provides several evaluation metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC Curve**: Visualization of true positive rate vs false positive rate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 