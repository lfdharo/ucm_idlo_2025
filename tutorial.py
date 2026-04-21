"""
TUTORIAL: Speaker Identification Using Deep Learning Models
============================================================

This tutorial shows you how to use the speaker identification system

Author: Luis F. D'Haro
Date: Apr 7, 2026
Course: Identificación de Locutores - Máster Lingüística y Tecnologías - UCM/UPM
"""

# ============================================================================
# STEP 1: IMPORT THE TOOLS
# ============================================================================
# These imports give you access to all the speaker identification functions

from easy_interface import SimpleSpeakerID
from tts_option import TTSOption
from visualization import plot_metrics, plot_by_speaker, compare_model_performance
from attention_visualization import (
    visualize_spectrogram, 
    visualize_speaker_comparison, 
    visualize_temporal_focus,
    AttentionVisualizer
)
from speaker_similarity_analysis import SpecificSpeakerComparison, compare_speakers
from typing import Optional
import logging

# Reduce logging verbosity if you want a cleaner output
logging.basicConfig(level=logging.WARNING)


# ============================================================================
# STEP 2: INITIALIZE THE SYSTEM
# ============================================================================
# This creates the speaker identification system with your chosen model

def basic_usage_example(model_name='wavLM', file_to_identify='./test/SPK1_A.wav'):
    """Example 1: Basic speaker identification"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Speaker Identification")
    print("="*70 + "\n")
    
    # Create the system with wavLM model
    system = SimpleSpeakerID(model_name=model_name, verbose=True)
    
    # Show what speakers are enrolled
    enrolled_speakers = system.list_enrolled_speakers()
    print(f"Enrolled speakers: {enrolled_speakers}\n")
    
    # Verify a single speaker
    print(f"Identifying a test audio: {file_to_identify}\n")
    result = system.identify(file_to_identify)
    
    # Print the result nicely
    print(f"✓ Speaker identified: {result['speaker_id']}")
    print(f"✓ Confidence: {result['confidence']:.2%}")
    print(f"✓ Above threshold: {result['is_match']}")
    print(f"✓ Message: {result['message']}\n")


def batch_processing_example(model_name='wavLM', test_files=None):
    """Example 2: Identify multiple speakers at once"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing (Multiple Audio Files)")
    print("="*70 + "\n")
    
    system = SimpleSpeakerID(model_name=model_name, verbose=False)
    
    # List of audio files to identify
    if test_files is None:
        test_files = [
            './test/SPK1_A.wav',
            './test/SPK2_A.wav',
            './test/SPK3_A.wav',
            './test/SPK4_A.wav'
        ]
    
    print(f"Processing {len(test_files)} audio files...\n")
    results = system.identify_batch(test_files)
    
    # Print summary
    print("\nRESULTS:")
    print("-" * 70)
    print(f"{'File':<25} | {'Speaker':<12} | {'Confidence':<12} | {'Match':<8}")
    print("-" * 70)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['file']:<25} | {result['speaker_id']:<12} | "
                  f"{result['confidence']:>10.2%} | {str(result['is_match']):<8}")
    
    print("-" * 70 + "\n")


def evaluation_example(model_name='wavLM'):
    """Example 3: Evaluate the system on all test files"""
    print("\n" + "="*70)
    print("EXAMPLE 3: System Evaluation")
    print("="*70 + "\n")
    
    system = SimpleSpeakerID(model_name=model_name, verbose=False)
    
    print("Evaluating system on all test files...")
    print("(This may take a few minutes)\n")
    
    # Get performance metrics
    metrics = system.evaluate()
    
    # The system prints a summary automatically, but you can also access the data
    print("\nYou can access individual metrics like this:")
    print(f"  accuracy = {metrics['accuracy']:.2%}")
    print(f"  precision = {metrics['precision']:.2%}")
    print(f"  recall = {metrics['recall']:.2%}")
    print(f"  f1_score = {metrics['f1_score']:.2%}\n")
    
    # Plot the results
    plot_metrics(metrics, title=f"Performance with {system.model_name} Model")
    plot_by_speaker(metrics['by_speaker'])


def threshold_experiment(model_name='wavLM'):
    """Example 4: See how threshold affects results"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Threshold Optimization")
    print("="*70 + "\n")
    
    print("Testing different thresholds to find the best one...\n")
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_by_threshold = {}
    
    for threshold in thresholds:
        system = SimpleSpeakerID(model_name=model_name, threshold=threshold, verbose=False)
        metrics = system.evaluate()
        results_by_threshold[f'Thr. {threshold}'] = metrics
    
    # Plot comparison
    compare_model_performance(results_by_threshold, metric='f1_score',
                             title='Effect of Threshold on Performance')


def visualize_audio_example(audio_file_test='./test/SPK1_A.wav', audio_file_enrollment='./enrollment/SPK1_0001.wav'):
    """Example 5: Visualize audio spectrograms"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Audio Visualization")
    print("="*70 + "\n")
    
    print("Displaying spectrogram of a speaker...\n")
    
    # Show single spectrogram
    visualize_spectrogram(audio_file_test, title=f'Test file {audio_file_test}: Audio Spectrogram')
    
    # Compare two speakers (useful for forensic analysis!)
    print("\nComparing two speakers visually...\n")
    visualize_speaker_comparison(
        audio_file_enrollment,
        audio_file_test,
        title='Forensic Analysis: Speaker Comparison'
    )
    
    # Show temporal focus
    print("\nShowing which parts of the audio are important...\n")
    visualize_temporal_focus(audio_file_test, 
                            title='Which parts of the recording identify the speaker?')


def model_comparison_example():
    """Example 6: Compare performance of different models"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Model Comparison")
    print("="*70 + "\n")
    
    models_to_test = ['wavLM', 'SpeechBrain', 'Whisper']
    model_results = {}
    
    for model in models_to_test:
        print(f"\nEvaluating {model}...\n")
        system = SimpleSpeakerID(model_name=model, verbose=False)
        metrics = system.evaluate()
        model_results[model] = metrics
    
    # Compare all models
    compare_model_performance(model_results, metric='accuracy',
                            title='Model Comparison: Accuracy')


def advanced_metrics_example(model_name='wavLM'):
    """Example 7: Calculate and display advanced metrics (EER, DET, ROC curves)"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Advanced Metrics for Forensic Analysis")
    print("="*70 + "\n")
    
    system = SimpleSpeakerID(model_name=model_name, verbose=False)
    
    # Calculate Equal Error Rate (EER)
    print("Calculating Equal Error Rate (EER)...")
    eer_threshold, eer_value = system.calculate_eer()
    print(f"✓ EER: {eer_value:.4f} at threshold {eer_threshold:.4f}")
    
    # Get detailed EER metrics
    eer_metrics = system.get_eer_metrics()
    print(f"\nAt EER threshold ({eer_threshold:.4f}):")
    print(f"  FAR (False Acceptance Rate): {eer_metrics['far']:.4f}")
    print(f"  FRR (False Rejection Rate): {eer_metrics['frr']:.4f}")
    
    # Plot ROC Curve (Receiver Operating Characteristic)
    print("\n\nPlotting ROC Curve...")
    print("(Shows trade-off between True Positive Rate and False Positive Rate)")
    system.plot_roc_curve(save_to='results/roc_curve.png')
    
    # Plot DET Curve (Detection Error Tradeoff) - Important for forensics!
    print("\n\nPlotting DET Curve...")
    print("(Shows trade-off between FAR and FRR on logarithmic scale)")
    system.plot_det_curve(save_to='results/det_curve.png')


def per_speaker_metrics_example(model_name='wavLM'):
    """Example 8: Detailed per-speaker metrics with manual and optimal threshold comparison"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Per-Speaker Detailed Metrics")
    print("="*70 + "\n")
    
    system = SimpleSpeakerID(model_name=model_name, verbose=False)
    
    # Get manual threshold (current system setting)
    manual_threshold = system.threshold
    
    print("Evaluating system on all test files...\n")
    metrics = system.evaluate()
    
    # Print detailed per-speaker metrics
    print("\nDETAILED PER-SPEAKER PERFORMANCE:")
    print("-" * 90)
    print(f"{'Speaker':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 90)
    
    for speaker, perf in metrics['by_speaker'].items():
        print(f"{speaker:<12} | {perf.get('accuracy', 0):>8.2%} | "
              f"{perf.get('precision', 0):>8.2%} | {perf.get('recall', 0):>8.2%} | "
              f"{perf.get('f1_score', 0):>8.2%}")
    
    print("-" * 90)
    
    # Also show FAR and FRR per speaker
    print("\n\nFAR/FRR PER SPEAKER (Important for forensic analysis):")
    print("-" * 60)
    print(f"{'Speaker':<12} | {'FAR':<15} | {'FRR':<15}")
    print("-" * 60)
    
    for speaker, perf in metrics['by_speaker'].items():
        far = perf.get('far', 0)
        frr = perf.get('frr', 0)
        print(f"{speaker:<12} | {far:>13.4f} | {frr:>13.4f}")
    
    print("-" * 60)
    
    # ============== THRESHOLD COMPARISON ============
    print("\n\n" + "="*70)
    print("THRESHOLD COMPARISON: Manual vs Optimal")
    print("="*70)
    
    # Calculate optimal threshold (EER)
    print("\nCalculating optimal threshold (EER - Equal Error Rate)...")
    eer_threshold, eer_value = system.calculate_eer()
    print(f"✓ Optimal EER threshold: {eer_threshold:.4f} (EER value: {eer_value:.4f})")
    
    # Get metrics at manual threshold
    print(f"\nGetting metrics at MANUAL threshold ({manual_threshold:.4f})...")
    manual_metrics = system.get_metrics_at_threshold(manual_threshold)
    
    # Get metrics at optimal threshold
    print(f"Getting metrics at OPTIMAL threshold ({eer_threshold:.4f})...")
    optimal_metrics = system.get_metrics_at_threshold(eer_threshold)
    
    # Display comparison
    print("\n" + "="*90)
    print(f"{'Metric':<20} | {'Manual Threshold':<20} | {'Optimal Threshold':<20} | {'Improvement':<15}")
    print(f"{'':20} | {f'({manual_threshold:.4f})':20} | {f'({eer_threshold:.4f})':20} | {'':15}")
    print("="*90)
    
    metrics_to_compare = [
        ('FAR', 'far'),
        ('FRR', 'frr'),
        ('F1 Score', 'f1'),
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('EER Value', 'far')  # Special case: EER is average of FAR and FRR
    ]
    
    for display_name, metric_key in metrics_to_compare:
        if metric_key == 'far' and display_name == 'EER Value':
            # EER is the average of FAR and FRR
            manual_val = (manual_metrics['far'] + manual_metrics['frr']) / 2
            optimal_val = (optimal_metrics['far'] + optimal_metrics['frr']) / 2
        else:
            manual_val = manual_metrics[metric_key]
            optimal_val = optimal_metrics[metric_key]
        
        # For FAR and FRR, lower is better; for F1, Accuracy, Precision, Recall, higher is better
        if metric_key in ['far', 'frr']:
            improvement = manual_val - optimal_val  # Positive means optimal is better
            improvement_str = f"↓ {improvement:.4f}" if improvement > 0 else f"↑ {abs(improvement):.4f}"
        else:
            improvement = optimal_val - manual_val  # Positive means optimal is better
            improvement_str = f"↑ {improvement:.4f}" if improvement > 0 else f"↓ {abs(improvement):.4f}"
        
        # Format based on metric type
        if metric_key in ['far', 'frr']:
            print(f"{display_name:<20} | {manual_val:>18.4f} | {optimal_val:>18.4f} | {improvement_str:>15}")
        else:
            print(f"{display_name:<20} | {manual_val:>18.2%} | {optimal_val:>18.2%} | {improvement_str:>15}")
    
    print("="*90)
    
    # Summary statistics
    print("\n\nDETAILED CONFUSION MATRICES:")
    print("-" * 70)
    print(f"At MANUAL threshold ({manual_threshold:.4f}):")
    print(f"  True Positives:  {manual_metrics['tp']:<4} | False Positives: {manual_metrics['fp']:<4}")
    print(f"  False Negatives: {manual_metrics['fn']:<4} | True Negatives:  {manual_metrics['tn']:<4}")
    
    print(f"\nAt OPTIMAL threshold ({eer_threshold:.4f}):")
    print(f"  True Positives:  {optimal_metrics['tp']:<4} | False Positives: {optimal_metrics['fp']:<4}")
    print(f"  False Negatives: {optimal_metrics['fn']:<4} | True Negatives:  {optimal_metrics['tn']:<4}")
    print("-" * 70 + "\n")


def confusion_matrix_example(model_name='wavLM', threshold: Optional[float] = None):
    """Example 9: Display confusion matrix for all test files
    
    A confusion matrix shows the distribution of:
    - True Positives (TP): Correct matches
    - True Negatives (TN): Correct rejections
    - False Positives (FP): Incorrect matches (Type I error)
    - False Negatives (FN): Incorrect rejections (Type II error)
    
    Useful for understanding the system's error distribution and trade-offs.
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Confusion Matrix Visualization")
    print("="*70 + "\n")
    
    system = SimpleSpeakerID(model_name=model_name, verbose=False)
    
    # Use provided threshold or system default
    if threshold is None:
        threshold = system.threshold
        print(f"Using default system threshold: {threshold:.4f}\n")
    else:
        print(f"Using custom threshold: {threshold:.4f}\n")
    
    # Display confusion matrix at the specified threshold
    print("Generating confusion matrix for all test files...\n")
    system.display_confusion_matrix(
        threshold=threshold,
        save_to='results/confusion_matrix.png'
    )
    
    print("\n✓ Confusion matrix displayed and saved to results/confusion_matrix.png")
    print("\nInterpretation:")
    print("  - Top-left (True Negatives): Correctly rejected non-matches")
    print("  - Top-right (False Positives): Incorrectly accepted non-matches (security risk)")
    print("  - Bottom-left (False Negatives): Incorrectly rejected matches (usability issue)")
    print("  - Bottom-right (True Positives): Correctly accepted matches")
    print("\n")


def real_model_attention_example(model_name='wavLM', audio_file='./test/SPK1_A.wav'):
    """Example 10: Visualize REAL attention weights from the model (not fallback).
    
    Shows actual attention from the neural network during speaker identification.
    Demonstrates where the model "looks" in the audio.
    
    Supported models:
    - wavLM: Recommended, fast
    - Whisper: Alternative choice
    - unispeech: High accuracy alternative
    - xlsr: Multilingual option
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: Real Model Attention Visualization")
    print("="*70 + "\n")
    
    print(f"Model: {model_name}")
    print(f"Audio file: {audio_file}\n")
    
    visualizer = AttentionVisualizer(model_name=model_name)
    
    # Display temporal attention from the actual model
    print("Extracting real attention weights from the model...")
    visualizer.display_model_attention_temporal(
        audio_file,
        model_name=model_name,
        title=f"{model_name} Actual Attention Weights Over Time",
        save_to=f'results/{model_name}_attention_temporal.png'
    )
    
    # Display attention heatmap (2D self-attention pattern)
    print("\nExtracting attention heatmap (how model attends to different time frames)...")
    visualizer.display_attention_heatmap(
        audio_file,
        model_name=model_name,
        title=f"{model_name} Self-Attention Heatmap",
        save_to=f'results/{model_name}_attention_heatmap.png'
    )
    
    print(f"\n✓ Real attention visualization complete for {model_name}")


def speaker_similarity_comparison_example():
    """Example 11: Compare two speakers and find similar spectral regions.
    
    For forensic analysis:
    - Finds the closest matching enrollment file for a test file
    - Compares their spectrograms
    - Shows which parts of the audio are similar
    - Highlights differences that could be due to:
      * Different microphones
      * Emotional state
      * Health conditions
      * Recording environment
    """
    print("\n" + "="*70)
    print("EXAMPLE 11: Speaker Similarity Analysis (Forensic Comparison)")
    print("="*70 + "\n")
    
    test_file = './test/SPK1_A.wav'
    enrollment_dir = './enrollment/'
    
    print(f"Test file: {test_file}")
    print(f"Enrollment directory: {enrollment_dir}\n")
    
    # Create comparison analyzer
    comparison = SpecificSpeakerComparison(model_name='wavLM')
    
    # Full analysis: find closest match and compare
    print("Step 1: Finding closest matching enrollment file...")
    results = comparison.analyze_and_compare(
        test_file,
        enrollment_dir,
        title_prefix="Forensic Speaker Comparison",
        save_dir='results/'
    )
    
    # Print detailed results
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"\nTest file: {results['test_file']}")
    print(f"Closest enrollment: {results['closest_enrollment']}")
    print(f"Embedding similarity: {results['embedding_similarity']:.4f}")
    print(f"Spectral similarity: {results['spectral_similarity']:.2%}")
    
    print("\nSimilar spectral regions (high agreement):")
    for start, end in results['similar_regions']:
        print(f"  Frames {start}-{end}")
    
    print("\nDifferent spectral regions (low agreement):")
    for start, end in results['different_regions'][:5]:  # Show first 5
        print(f"  Frames {start}-{end}")
    
    print("\n" + "="*70)
    print("Why analyze differences?")
    print("="*70)
    print("Different regions may indicate:")
    print("  • Speaker fatigue or emotional changes")
    print("  • Different microphone/recording equipment")
    print("  • Environmental noise at different times")
    print("  • Speech rate variations")
    print("  • Health-related voice changes")
    print("="*70 + "\n")


def advanced_data_augmentation_example(speaker_id='SPK1'):
    """
    Example 12: Advanced Data Augmentation
    
    Demonstrates advanced audio augmentation techniques for creating 
    robust training data. These techniques simulate real-world acoustic
    variations that speakers encounter.
    """
    print("\n" + "="*70)
    print("EXAMPLE 12: Advanced Data Augmentation")
    print("="*70)
    print("Purpose: Create training data variations")
    print("  - Spectral masking (simulates hearing loss/low bandwidth)")
    print("  - Temporal masking (simulates speech interruptions)")
    print("  - Moderate/aggressive combinations")
    print("  - Loudness variations")
    print("  - Frequency filtering\n")
    
    from advanced_augmentation import AdvancedAugmentation
    import os
    
    # Initialize augmentation
    aug = AdvancedAugmentation(main_path='./')
    
    print("📊 AUGMENTATION TECHNIQUES:\n")
    
    print("1️⃣  SPECTRAL MASKING - Simulates partial frequency loss (hearing loss, phone quality)")
    print("   Command: aug.augment_data('./test/', 'freqMask', speaker='SPK1')")
    print("   Use case: Test robustness to bandwidth limitations\n")
    
    print("2️⃣  TEMPORAL MASKING - Simulates speech interruptions (network dropout)")
    print("   Command: aug.augment_data('./test/', 'timeMask', speaker='SPK1')")
    print("   Use case: Test robustness to speech cuts\n")
    
    print("3️⃣  MODERATE COMBINATION - Balanced augmentation (realistic)")
    print("   Combines: Noise + Time-stretch + Pitch shift")
    print("   Command: aug.augment_data('./test/', 'moderate', speaker='SPK1')")
    print("   Use case: General robustness training\n")
    
    print("4️⃣  AGGRESSIVE COMBINATION - Heavy augmentation (extreme conditions)")
    print("   Combines: Heavy noise + 25% time-stretch + ±4 semitones pitch")
    print("   Command: aug.augment_data('./test/', 'aggressive', speaker='SPK1')")
    print("   Use case: Test system limits\n")
    
    print("5️⃣  LOUDNESS VARIATIONS - RMS-based scaling")
    print("   Command: signal_aug = signal * gain_linear")
    print("   Use case: Test dynamic range handling\n")
    
    print("6️⃣  HIGH-PASS FILTERING - Removes low frequencies (telephone simulation)")
    print("   Command: apply high-pass filter at 80Hz")
    print("   Use case: Test on phone-quality speech\n")
    
    print("7️⃣  LOW-PASS FILTERING - Removes high frequencies (speech compression)")
    print("   Command: apply low-pass filter at 3kHz")
    print("   Use case: Test on compressed/bandwidth-limited speech\n")
    
    # Example: Apply moderate augmentation
    print("="*70)
    print("✅ Applying moderate augmentation to SPK1 test files...")
    print("   This creates multiple variations of each test file\n")
    
    # Skip actual augmentation in demo mode
    # Uncomment to actually run:
    aug.augment_data('./test/', 'aggressive', speaker=speaker_id)
    
    print("Output files would be saved with suffixes:")
    print("  _gaussian.wav    (Gaussian noise)")
    print("  _timeStretch.wav (Speaking rate changed)")
    print("  _pitchShift.wav  (Pitch modified)")
    print("  _shift.wav       (Time-shifted)")
    print("  _moderate.wav    (Balanced combination)")
    print("  _aggressive.wav  (Heavy augmentation)")
    print("  _freqMask.wav    (Frequency masking)")
    print("  _timeMask.wav    (Temporal masking)\n")
    
    print("📚 FORENSIC INTERPRETATION:")
    print("  If your system successfully identifies the speaker across")
    print("  different augmentations, it's more robust to real-world")
    print("  acoustic variations found in forensic recordings.\n")
    print("="*70 + "\n")


def tts_spoofing_example(speaker_id='SPK1', reference_file=None, ref_text=None):
    """
    Example 13: TTS-Based Spoofing Robustness Evaluation
    
    Demonstrates voice cloning to evaluate if the speaker 
    identification system can detect when someone tries to spoof an enrolled 
    speaker's voice using TTS-based voice cloning.
    
    This is critical for forensic assessment: Can cloned speech fool the system?
    
    Available TTS backends:
    - 'qwen3tts': High-quality voice cloning (GPU recommended)
    - 'coqui': Flexible TTS - supports both regular TTS and voice cloning

    """
    print("\n" + "="*70)
    print("EXAMPLE 13: TTS-Based Spoofing Robustness Evaluation")
    print("="*70)
    print("Purpose: Test if speaker ID is robust against cloned speech")
    print("Method: Generate cloned speaker voice using TTS voice cloning\n")
    
    from tts_spoofing import TTSSpoofingGenerator
    from easy_interface import SimpleSpeakerID
    import os
    
    # Initialize with TTS spoofing generator with error handling
    print("Initializing TTS spoofing generator...")
    try:
        # Use 'simple' backend (auto-select: Qwen3TTS for cloning, CoquiTTS fallback)
        generator = TTSSpoofingGenerator(model='simple', use_gpu=True)
        if generator.tts is not None:
            backend_name = generator.tts.available_backend
            print(f"✓ TTS backend: {backend_name}\n")
        else:
            print("⚠ Warning: TTS not fully initialized\n")
    except Exception as e:
        print(f"✗ TTS initialization error: {e}")
        print("  Falling back to demonstration mode...\n")
        generator = None
    
    models_info = {
        'qwen3tts': 'Qwen3TTS voice cloning (best quality, GPU recommended)',
        'coqui': 'CoquiTTS (XTTS v2) - flexible, supports cloning and TTS',
        'simple': 'Auto-select best available backend (✓ recommended)'
    }
    
    print("="*70)
    print("1. AVAILABLE TTS BACKENDS:")
    print("="*70)
    for model, desc in models_info.items():
        print(f"  {model:12} → {desc}")
    
    print("\n" + "="*70)
    print("2. TEXT VARIATION DIFFICULTY LEVELS:")
    print("="*70)
    difficulties = {
        'easy': '3 utterances | Simple professional phrases',
        'medium': '6 utterances | Mix professional + casual (RECOMMENDED)',
        'hard': '8 utterances | Phonetically diverse + numbers',
        'expert': '15 utterances | Complete corpus, maximum coverage'
    }
    for diff, desc in difficulties.items():
        print(f"  {diff:8} → {desc}")
    
    print("\n" + "="*70)
    print("3. SAMPLE SPANISH TEXT TEMPLATES (auto-loaded):")
    print("="*70)
    samples = [
        "(Professional) Soy el locutor que necesita verificar su identidad.",
        "(Casual) Hola, ¿cómo estás? Me llamo locutor.",
        "(Numbers) Dígitos: uno, dos, tres, cuatro, cinco.",
        "(Diverse) La lluvia cayó sobre la montaña."
    ]
    for sample in samples:
        print(f"  • {sample}")
    
    print("\n" + "="*70)
    print("4. GENERATION EXAMPLE:")
    print("="*70)
    
    if reference_file is None:
        reference_file = f'./enrollment/{speaker_id}/{speaker_id}_0001.wav'
        ref_text = "ven aquí Watson"

    if not os.path.exists(reference_file):
        print(f"⚠️  Enrollment file not found: {reference_file}")
        print(f"     Expected: ./enrollment/SPEAKER/SPEAKER_*.wav")
        print("Skipping generation example.\n")
        return
    
    print(f"Reference enrollment: {reference_file}\n")
    print("Generating synthetic speaker...")
    
    metadata = generator.generate_synthetic_speaker(
        speaker_name=f'{speaker_id}_SPOOFED',
        speaker_wav=reference_file,
        output_dir='./synthetic_speakers/',
        language='spanish',
        difficulty='medium',
        ref_text=ref_text
    )
    
    print(f"\n✓ Generated: {metadata['generation_stats']['successful']} files")
    print(f"✓ Duration: {metadata['generation_stats']['total_duration']:.1f}s")
    
    print("\n" + "="*70)
    print("5. ROBUSTNESS EVALUATION (OPTIONAL):")
    print("="*70)
    print("To evaluate if your speaker ID system is fooled by synthetic speech:\n")
    
    print("Code example:")
    print("```python")
    print("# Load your speaker ID system")
    print("system = SimpleSpeakerID(model_name='wavLM')")
    print("system.enroll_speaker('SPK1', './enrollment/SPK1/')")
    print("")
    print("# Test against synthetic attempts")
    print("results = generator.evaluate_spoofing_robustness(")
    print("    speaker_id_func=system.identify,")
    print("    speaker_name='SPK1',")
    print("    synthetic_dir='./synthetic_speakers/SPK1_SPOOFED/synthetic/',")
    print("    enrollment_file='./enrollment/SPK1/SPK1_0001.wav'")
    print(")")
    print("")
    print("# Interpret results")
    print("sfar = results['false_accepts_rate']")
    print("if sfar > 0.10:")
    print("    print('⚠️  VULNERABLE to voice cloning')")
    print("elif sfar < 0.01:")
    print("    print('✅ ROBUST to voice cloning')")
    print("else:")
    print("    print('⚡ Moderate robustness')")
    print("```\n")
    print("="*70 + "\n")


# ============================================================================
# STEP 3: RUN AN EXAMPLE
# ============================================================================
# Uncomment the example you want to run, then execute this file

if __name__ == "__main__":
    
    # Run Example 1: Basic identification
    # basic_usage_example(file_to_identify='./test/SPK3_A.wav', model_name='wavLM')
    
    # Run Example 2: Batch processing
    # batch_processing_example(model_name='Whisper', test_files=[
    #    './test/SPK1_A.wav',
    #    './test/SPK1_A_gaussian.wav',])
    
    # Run Example 3: System evaluation
    # evaluation_example(model_name='wavLM')
    
    # Run Example 4: Threshold optimization
    # threshold_experiment(model_name='wavLM')
    
    # Run Example 5: Audio visualization (for forensic analysis!)
    # visualize_audio_example(audio_file_enrollment='./enrollment/SPK1/SPK1_0001.wav', audio_file_test='./test/SPK1_A.wav')
    
    # Run Example 6: Model comparison
    # model_comparison_example()
    
    # Run Example 7: Advanced metrics (EER, DET, ROC curves)
    # advanced_metrics_example(model_name='wavLM')
    
    # Run Example 8: Per-speaker detailed metrics (with manual vs optimal threshold comparison)
    # per_speaker_metrics_example(model_name='wavLM')
    
    # Run Example 9: Confusion Matrix visualization
    # Shows True Positives, True Negatives, False Positives, False Negatives
    # confusion_matrix_example(model_name='wavLM', threshold=0.5)
    
    # ====================================================================
    # DNN-based Model Attention & Speaker Similarity Analysis
    # ====================================================================
    
    # Run Example 10: Use Model Attention Weights to visualize which parts of the audio the model focuses on
    # Shows actual neural network attention, not fallback spectrogram
    # Supported models: 'wavLM', 'Whisper', 'unispeech', 'xlsr'
    # real_model_attention_example(model_name='wavLM', audio_file='./test/SPK1_A.wav')
    
    # Run Example 11: SPEAKER SIMILARITY ANALYSIS
    # For forensic analysis: compares two speakers and finds similar regions
    # speaker_similarity_comparison_example()
    
    # ====================================================================
    # Advanced Data Augmentation & TTS Spoofing
    # ====================================================================
    
    # Run Example 12: Advanced data augmentation
    # Demonstrates spectral masking, temporal masking, filtering, etc.
    # advanced_data_augmentation_example(speaker_id='SPK1')
    
    # Run Example 13: TTS-based spoofing robustness evaluation
    # Tests if system can detect voice cloning attacks
    # Note: Comment in to run (requires TTS models and may need GPU)
    tts_spoofing_example(speaker_id='SPK1', reference_file='./enrollment/SPK1/SPK1_0001.wav', ref_text="ven aquí Watson")
   
    from tts_option import TTSOption
    # Create TTS clone voice
    tts_models = TTSOption()
    tts_models.create_tts_clone(message="Esto es una prueba de voz clonada.", model_voice="./test/SPK1_A.wav", output_file_path="./test/SPK1_A_CLONED.wav", language="spanish")
    basic_usage_example(file_to_identify='./test/SPK1_A_CLONED.wav', model_name='wavLM')

    # Create TTS voice
    tts_models.create_tts(message="Esto es una prueba de voz sintética.", output_file_path="./test/SPK1_TTS.wav")
    basic_usage_example(file_to_identify='./test/SPK1_TTS.wav', model_name='wavLM')

    print("\n" + "="*70)
    print("Tutorial completed!")
    print("="*70 + "\n")
