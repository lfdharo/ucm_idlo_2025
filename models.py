import torch
import numpy as np
import logging
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor
from transformers import WavLMForXVector
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.encoders import WaveformEncoder
from transformers import WhisperProcessor, WhisperModel
from vector_embedding import exctract_vector_embedding
from deepspeaker import DeepSpeakerModel

class ModelFactory:
    """Factory class to create and manage different speaker verification models."""
    
    @staticmethod
    def create_model(model_name: str, device: Optional[str] = None) -> tuple:
        """Create a model and its feature extractor based on the model name.
        
        Args:
            model_name (str): Name of the model to create
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
            
        Returns:
            tuple: (model, feature_extractor)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        logger = logging.getLogger(__name__)
        logger.info(f"Creating model {model_name} on device {device}")
        
        if model_name == 'wavLM':
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
            model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)

            
        elif model_name == 'SpeechBrain':
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb", 
                savedir="pretrained_models/spkrec-xvect-voxceleb",
                run_opts={"device": device}
            )
            feature_extractor = None

        elif model_name == 'whisper':
            feature_extractor = None
            model = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

        elif model_name == 'deepspeaker':
            model = DeepSpeakerModel()
            model.rescnn.load_weights("./ResCNN_triplet_training_checkpoint_265.h5")
            feature_extractor = None
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        return model, feature_extractor
        
    @staticmethod
    def extract_embedding(model_name: str, audio_file: str, model: object, 
                         feature_extractor: Optional[object] = None) -> np.ndarray:
        """Extract speaker embedding using the specified model.
        
        Args:
            model_name (str): Name of the model
            audio_file (str): Path to audio file
            model (object): Model instance
            feature_extractor (object, optional): Feature extractor instance
            
        Returns:
            np.ndarray: Speaker embedding vector
        """
        return exctract_vector_embedding(audio_file, model, model_name, feature_extractor) 
