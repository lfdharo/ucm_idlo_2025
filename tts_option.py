from TTS.api import TTS
import torch

class TTSOption:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_clone = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.tts = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(device)

    def create_tts_clone(self, message, model_voice, language, output_file_path):
        # generate speech by cloning a voice using default settings
        self.tts_clone.tts_to_file(text=message,
                file_path=output_file_path,
                speaker_wav=model_voice,
                language=language)

    def create_tts(self, message, output_file_path):
        # generate speech by cloning a voice using default settings
        self.tts.tts_to_file(text=message,
                file_path=output_file_path)
