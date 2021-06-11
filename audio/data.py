import torch
import os
import pathlib
from glob import glob
import wave
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

DATA_DIR = "data/mini_speech_commands"


def read_wav_file(file):
    f = wave.open(file, "rb")
    n_frames = f.getnframes()
    str_data = f.readframes(n_frames)
    f.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    return wave_data

def get_spectrogram(w):
    zero_pad_len = 16000 - w.shape[0]
    w = np.pad(w, (0, zero_pad_len))
    _, _, spec = signal.stft(w, nfft=255, nperseg=255)
    spec = np.abs(spec)
    return spec

def normalize_data(spec):
    row = spec.shape[0]
    col = spec.shape[1]
    n = row * col
    mean = np.sum(spec) / n
    std = np.sqrt(np.sum(np.square(spec - mean)) / n)
    return (spec - mean) / (std + 1e-7)


def preprocess_data(w):
    spec = get_spectrogram(w)
    # spec.resize(32, 32)
    spec = np.log(spec + 1e-7)
    # spec = normalize_data(spec)
    spec = spec.reshape(1, spec.shape[0], spec.shape[1])
    return spec


class AudioDataset(Dataset):

    def __init__(self, data_dir=DATA_DIR):
        self.label2idx, self.data = self.load_data(data_dir)
        self.size = len(self.data)
        self.shape = self.data[0][0].shape

    def load_data(self, data_dir):
        audio_files = glob(data_dir + "/*/*")
        label_dirs = glob(data_dir + "/*")
        label_names = [_dir.split('\\')[-1] for _dir in label_dirs]
        label_names = [l for l in label_names if l != "README.md"]
        label2idx = {label: idx for idx, label in enumerate(label_names)}
        waves = [read_wav_file(file) for file in audio_files]
        spectrograms = [preprocess_data(w) for w in waves]
        labels = [label2idx[file.split('\\')[-2]] for file in audio_files]
        data = [(spectrograms[idx], labels[idx]) for idx in range(len(labels))]

        return label2idx, data


class Dataloader(object):
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.size = len(self.data)
        self.steps_per_epoch = self.size // self.batch_size
        self.shape = self.data[0][0].shape
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_data(self):
        batch_idx_list = np.random.choice(self.size, self.batch_size)
        batch_data = torch.zeros((self.batch_size,) + self.data[0][0].shape).to(self.device)
        batch_label = torch.zeros(self.batch_size, dtype=torch.int64).to(self.device)
        for i, idx in enumerate(batch_idx_list):
            batch_data[i] = torch.from_numpy(self.data[idx][0])
            batch_label[i] = self.data[idx][1]
        return batch_data, batch_label


if __name__ == '__main__':
    dataset = AudioDataset()
    train_idx_list = np.random.choice(dataset.size, int(dataset.size * 0.8))
    train_data = [dataset.data[idx] for idx in train_idx_list]
    test_data = [dataset.data[idx] for idx in range(dataset.size) if idx not in train_idx_list]
    train_dataloader = Dataloader(train_data)
    test_dataloader = Dataloader(test_data)

    train_batch_data = train_dataloader.get_data()
    print(len(train_batch_data))
    print([spec.shape for spec, _ in train_batch_data])
    print([lab for _, lab in train_batch_data])
