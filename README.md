# 🎙️ Speaker Identification System for Linguistic Research

A deep learning-based speaker identification system designed for **linguistic students** and **forensic phonetics** research. Easily identify speakers, analyze model decisions, and visualize audio characteristics.

**👉 START HERE:** Read [STUDENT_GUIDE.md](STUDENT_GUIDE.md) for a comprehensive tutorial!

---

## ⚡ Quick Start (30 seconds)

```python
from easy_interface import SimpleSpeakerID
from visualization import plot_metrics

# Initialize system
system = SimpleSpeakerID(model_name='wavLM')

# Identify a speaker
result = system.identify('audio.wav')
print(f"Speaker: {result['speaker_id']} ({result['confidence']:.0%})")

# Evaluate performance
metrics = system.evaluate()
plot_metrics(metrics)
```

---

## ✨ Key Features

### 🎯 Easy-to-Use Interface
- **SimpleSpeakerID** class handles all complexity
- Works with minimal Python knowledge
- Built-in error handling and helpful messages

### 🧠 Multiple AI Models
- **WavLM**: Fast, high accuracy (recommended)
- **SpeechBrain**: Great with noisy audio
- **Whisper**: Low resource requirements

### 📊 Comprehensive Visualization
- Performance metrics (accuracy, precision, recall, F1)
- Per-speaker analysis
- Model comparison plots
- Spectrogram visualization for audio analysis

### 🔍 Forensic Analysis Tools
- **Attention visualization**: See where the model "looks" in the audio
- **Temporal focus**: Which time regions matter for identification
- **Frequency analysis**: Which frequencies are important
- **Speaker comparison**: Visual comparison of two voices

### 📈 Evaluation & Metrics
- Overall performance metrics
- Per-speaker performance analysis
- Confusion matrices
- Threshold optimization
- Batch processing support

---

## 📦 Installation

### Step 1: Setup Environment
```bash
# Clone/download the project
cd speaker-identification/

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
# For local machine
pip install -r requirements_local.txt

# For Google Colab
pip install -r requirements_colab.txt
```

### Step 3: Verify Installation
```python
from easy_interface import SimpleSpeakerID
system = SimpleSpeakerID(model_name='wavLM', verbose=True)
print("✓ Installation successful!")
```

---

## 📁 Project Structure

```
project/
├── easy_interface.py              # ← Start here! Simplified API
├── visualization.py               # Easy plotting functions
├── attention_visualization.py      # Forensic analysis tools
├── tutorial.py                    # Working examples
├── STUDENT_GUIDE.md               # Complete documentation
│
├── models.py                      # AI model loading
├── faiss_class.py                 # Speaker matching
├── vector_embedding.py            # Audio processing
├── evaluation.py                  # Metrics & analysis
├── data_augmentation.py           # Audio augmentation
├── tts_option.py                  # Text-to-speech
│
├── enrollment/                    # Training data
│   ├── SPK1/, SPK2/, SPK3/       # Speaker folders
│   └── enrollment_index_*         # Cached indices
├── test/                          # Test audio files
└── results/                       # Output plots & reports
```

---

## 🚀 Usage Examples

### Example 1: Identify a Single Speaker
```python
from easy_interface import SimpleSpeakerID

system = SimpleSpeakerID()
result = system.identify('./test/mystery_audio.wav')

print(f"Identified as: {result['speaker_id']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Match: {result['is_match']}")
```

### Example 2: Batch Processing
```python
files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = system.identify_batch(files)

for r in results:
    print(f"{r['file']}: {r['speaker_id']} ({r['confidence']:.0%})")
```

### Example 3: Evaluate Entire System
```python
metrics = system.evaluate()
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Visualize results
from visualization import plot_metrics, plot_by_speaker
plot_metrics(metrics)
plot_by_speaker(metrics['by_speaker'])
```

### Example 4: Forensic Audio Analysis
```python
from attention_visualization import visualize_spectrogram, visualize_speaker_comparison

# View audio spectrogram
visualize_spectrogram('./test/SPK1_A.wav')

# Compare two speakers visually
visualize_speaker_comparison('speaker1.wav', 'speaker2.wav')
```

### Example 5: Try Different Models
```python
from visualization import compare_model_performance

results = {}
for model in ['wavLM', 'SpeechBrain']:
    system = SimpleSpeakerID(model_name=model)
    results[model] = system.evaluate()

compare_model_performance(results, metric='accuracy')
```

---

## 📖 Complete Documentation

### For Students
- **[STUDENT_GUIDE.md](STUDENT_GUIDE.md)** - Comprehensive guide with examples
- **tutorial.py** - 6 complete working examples (run it!)
- Code comments - Every function is thoroughly documented

### For Developers
- [models.py](models.py) - Model factory and implementations
- [faiss_class.py](faiss_class.py) - FAISS-based similarity search
- [evaluation.py](evaluation.py) - Metrics and analysis
- [vector_embedding.py](vector_embedding.py) - Audio feature extraction

---

## 🎯 Available Models

| Model | Speed | Accuracy | Best Use Case |
|-------|-------|----------|---------------|
| **WavLM** | ⚡⚡⚡ (Fast) | 95%+ | General purpose, clean audio |
| **SpeechBrain** | ⚡⚡ (Medium) | 93%+ | Noisy environments, robust |
| **Whisper** | ⚡ (Slower) | 90%+ | Low resources, multilingual |

---

## 🔧 Advanced Features

### Threshold Optimization
Find the best decision threshold for your use case:
```python
system.set_threshold(0.6)  # More conservative
metrics = system.evaluate()
```

### Multiple Metrics
```python
metrics = system.evaluate()
print(f"Accuracy:  {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")  # Important for forensics!
print(f"Recall:    {metrics['recall']:.2%}")
print(f"F1 Score:  {metrics['f1_score']:.2%}")
```

### Transform Results to DataFrame
```python
import pandas as pd
results = system.identify_batch(files)
df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
```

---

## 🛟 Troubleshooting

### "File not found" errors
Make sure enrollment and test directories exist:
```python
import os
os.makedirs('enrollment', exist_ok=True)
os.makedirs('test', exist_ok=True)
```

### Out of memory?
Use CPU instead of GPU:
```python
import torch
torch.cuda.is_available = lambda: False
```

### Low accuracy?
1. Check audio quality (16 kHz, mono, clean)
2. Use more enrollment samples per speaker
3. Try different models
4. Adjust threshold

See [STUDENT_GUIDE.md](STUDENT_GUIDE.md#troubleshooting) for more solutions.

---

## 📚 Learning Resources

### Key Concepts
- **Speaker Embedding**: Fixed-size vector representing voice characteristics
- **Similarity Score**: Distance between embeddings (lower = same speaker)
- **Threshold**: Minimum confidence needed for a positive match
- **Precision vs Recall**: Trade-off in decision making

### Related Papers (Optional Reading)
- wavLM: Large-Scale Self-Supervised Pre-training for Speech
- ECAPA-TDNN: Speaker Verification with Channel Attention
- RawNet3: Speaker Identification with Deep Raw Waveform Networks

### Linguistic Resources
- *Forensic Phonetics* - Expert analysis of speaker identification
- *Automatic Speaker Recognition* - Technical foundations
- Speech databases: VoxCeleb, TIMIT, Common Voice

---

## 🎓 Student Project Ideas

1. **Forensic Case Study**: Analyze a questioned voice against multiple known speakers
2. **Robustness Testing**: How does noise/emotion/accent affect identification?
3. **Model Comparison**: Which model works best for different types of audio?
4. **Feature Analysis**: Which acoustic features matter most?
5. **Cross-Language Study**: How does language affect speaker identification?

---

## 📝 Citation

If you use this system in your research:

```bibtex
@misc{speaker_id_2026,
  title={Speaker Identification System for Linguistic Research},
  author={D'Haro, Luis F.},
  note={Máster Lingüística y Tecnologías - UCM/UPM},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the APACHE License - see [LICENSE](LICENSE) for details.

---

## ❓ Questions?

1. Check [STUDENT_GUIDE.md](STUDENT_GUIDE.md)
2. Review code comments (all documented!)
3. Run [tutorial.py](tutorial.py) for working examples
4. Check the docstrings: `help(SimpleSpeakerID.identify)`

---

## 👨‍🏫 Course

**Identificación de Locutores**
- Máster Lingüística y Tecnologías
- UCM/UPM
- Prof. Luis F. D'Haro

Happy researching! 🎙️ 