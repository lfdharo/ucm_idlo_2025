import torch
import warnings
import logging

logger = logging.getLogger(__name__)

# Monkey patch torch.load for PyTorch 2.6+ compatibility with older model checkpoints
_original_torch_load = torch.load
def torch_load_with_weights_only_false(*args, **kwargs):
    """
    Wrapper for torch.load that forces weights_only=False for backward compatibility.
    
    PyTorch 2.6+ changed default from weights_only=False to weights_only=True.
    This breaks loading old models that use defaultdict and other non-standard types.
    This patch forces weights_only=False to load legacy models correctly.
    """
    # FORCE weights_only=False to allow defaultdict and other legacy types
    kwargs['weights_only'] = False
    
    # Suppress the weights_only warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _original_torch_load(*args, **kwargs)

torch.load = torch_load_with_weights_only_false


class TTSOption:
    def __init__(self):
        """Initialize TTSOption with lazy TTS loading (deferred to first use)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_clone = None
        self.tts = None
        self._initialized = False
    
    def _init_tts(self):
        """Lazy initialize TTS models (deferred to avoid early CUDA errors)"""
        if self._initialized:
            return
        
        try:
            from TTS.api import TTS
            
            logger.info(f"Initializing TTS on device: {self.device}")
            self.tts_clone = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            self.tts = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(self.device)
            self._initialized = True
            logger.info("TTS initialization successful")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {str(e)}")
            raise

    def create_tts_clone(self, message, model_voice, language, output_file_path):
        """Generate speech by cloning a voice using default settings"""
        self._init_tts()  # Lazy initialization on first use
        self.tts_clone.tts_to_file(text=message,
                file_path=output_file_path,
                speaker_wav=model_voice,
                language=language)

    def create_tts(self, message, output_file_path):
        """Generate speech using default TTS model"""
        self._init_tts()  # Lazy initialization on first use
        self.tts.tts_to_file(text=message,
                file_path=output_file_path)


if __name__ == "__main__":
    tts_option = TTSOption()
    message = "Hola, ¿cómo estás?"
    output_file_path = "output.wav"
    tts_option.create_tts(message, output_file_path)
    print(f"Generated speech saved to {output_file_path}")
    output_file_path = "output2.wav"
    tts_option.create_tts_clone(message, "./enrollment/SPK1/SPK1_0001.wav", "es", output_file_path)