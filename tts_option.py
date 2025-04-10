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


if __name__ == "__main__":
    tts_option = TTSOption()
    message = "Hola, ¿cómo estás?"
    output_file_path = "output.wav"
    tts_option.create_tts(message, output_file_path)
    print(f"Generated speech saved to {output_file_path}")
    output_file_path = "output2.wav"
    tts_option.create_tts_clone(message, "./enrollment/SPK1/SPK1_0001.wav", "es", output_file_path)