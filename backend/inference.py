import torch
import numpy as np
import librosa
from tacotron2.model import Tacotron2
from waveglow.denoiser import Denoiser
from text import text_to_sequence
import time
import threading

class Synthesizer:
    def __init__(self, tacotron2_path, waveglow_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        self.lock = threading.Lock()
        self.load_models(tacotron2_path, waveglow_path)
    
    def load_models(self, tacotron2_path, waveglow_path):
        with self.lock:
            if self.models_loaded:
                return
            
            # Load Tacotron2
            self.tacotron2 = Tacotron2().to(self.device)
            self.tacotron2.load_state_dict(torch.load(tacotron2_path)['state_dict'])
            self.tacotron2.eval()
            
            # Load WaveGlow
            waveglow = torch.load(waveglow_path, map_location=self.device)['model']
            self.waveglow = waveglow.remove_weightnorm(waveglow)
            self.waveglow.eval()
            
            # Create denoiser
            self.denoiser = Denoiser(self.waveglow)
            self.models_loaded = True

    def synthesize(self, text, pitch_shift=0, speed_factor=1.0):
        with self.lock:
            start_time = time.time()
            
            # Preprocess text
            sequence = text_to_sequence(text, ['english_cleaners'])
            sequence = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate mel-spectrogram
            with torch.no_grad():
                mel_outputs, _, _ = self.tacotron2.inference(sequence)
                audio = self.waveglow.infer(mel_outputs, sigma=0.666)
                audio = self.denoiser(audio, strength=0.01)[0]
            
            # Convert to numpy
            audio = audio.cpu().numpy()
            audio = audio.astype(np.float32)
            
            # Apply audio transformations
            if speed_factor != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            if pitch_shift != 0:
                audio = librosa.effects.pitch_shift(
                    audio, 
                    sr=22050, 
                    n_steps=pitch_shift,
                    res_type='kaiser_fast'
                )
            
            # Peak normalization
            audio = audio / np.max(np.abs(audio))
            
            return audio
