import librosa
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset
import torch
import random

class RawMelDataset(Dataset):
    def __init__(self, files, audio_length, config):
        self.audio_length = audio_length
        self.config = config
        self.files = files
        
    
    def _compute_mel_spec(self, y):
        return librosa.feature.melspectrogram(y=y,
                                              n_mels=self.config.n_mels,
                                              sr=self.config.sample_rate, 
                                              n_fft=self.config.n_fft, 
                                              hop_length=self.config.hop_length,
                                              win_length=self.config.win_length,
                                              power=self.config.power,
                                              fmin=self.config.fmin,
                                              fmax=self.config.fmax)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        y, sr = librosa.load(self.files[index])
        if sr != self.config.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.config.sample_rate)

        max_v = np.abs(y).max()
        if max_v >= 1.0:
            y = y*(0.998/max_v)

        # two cases 
        if len(y) < self.audio_length:
            y = np.pad(y, (0, self.audio_length - len(y)))
        else:
            bidx = random.randint(0, len(y) - self.audio_length)
            y = y[bidx:bidx+self.audio_length]
        
        # encode
        melspecs = torch.tensor(self._compute_mel_spec(y))
        y = torch.tensor(y).long()
        return (y, melspecs), y



def load_ljspeech_dataset(config):
    """
    Returns:
        tuple: (train dataset, test dataset)
    """
    files = glob(os.path.join(config.dataset_path, f"{config.lj_folder_name}/wavs/*.wav"))
    test_len = int(len(files)/20)
    # train, test = random_split(files, [len(files)-test_len, test_len])
    train, test = files[:len(files)-test_len], files[-test_len:]
    return RawMelDataset(train, config.lj_train_audio_length, config), RawMelDataset(test, config.lj_train_audio_length, config)

