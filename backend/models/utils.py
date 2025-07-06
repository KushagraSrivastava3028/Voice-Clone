import torch
import numpy as np
import librosa

def load_wav_to_torch(full_path, sr=22050):
    data, sr = librosa.load(full_path, sr=sr)
    return torch.FloatTensor(data.astype(np.float32)), sr

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask
