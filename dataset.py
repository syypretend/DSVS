import json
import os
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler
from PIL import Image
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tool.triplet2IVT import triplet2IVT


class CholecVideo(Dataset):
    def __init__(self, data_dir, triplet_dir, phase_dir, video_names, sequence_length, transform=None, batch_size=None):
        self.data_dir = data_dir
        self.triplet_dir = triplet_dir
        self.phase_dir = phase_dir
        self.video_names = video_names
        self.seq_len = sequence_length
        self.transform = transform
        self.labels = {}
        self.batch_size = batch_size
        self.get_labels()
        # the length of origin videos
        self.n_frames = []
        for name in self.video_names:
            self.n_frames.append(len(os.listdir(os.path.join(data_dir, name))))
        self.size = sum(self.n_frames)

    def __getitem__(self, index):
        frame_idx, v_name, img_name = self.get_idx(index)
        fpath = os.path.join(self.data_dir, v_name, img_name)
        phase = self.get_phase(frame_idx, v_name)
        if self.transform is None:
            x = torch.from_numpy(np.array(Image.open(fpath)))
        else:
            x = torch.from_numpy(np.array(self.transform(Image.open(fpath))))

        _, y = self.labels[v_name][frame_idx]  # y is a list. len(y) = 100
        y = torch.from_numpy(np.array(y))
        return frame_idx, v_name, x, y, phase

    def __len__(self):
        self.size = sum(self.n_frames)
        return self.size

    def get_phase(self, idx, video_name):
        a = torch.zeros(7)
        df = pd.read_csv(os.path.join(self.phase_dir, video_name + ".txt"), sep="\t")
        if idx * 25 + 2 > len(df):
            b = torch.tensor(6)
            a[6] = 1
            return b
        else:
            if df["Phase"][idx * 25 + 2] == "Preparation":
                b = torch.tensor(0)
                a[0] = 1
                return b
            elif df["Phase"][idx * 25 + 2] == "CalotTriangleDissection":
                b = torch.tensor(1)
                a[1] = 1
                return b
            elif df["Phase"][idx * 25 + 2] == "ClippingCutting":
                b = torch.tensor(2)
                a[2] = 1
                return b
            elif df["Phase"][idx * 25 + 2] == "GallbladderDissection":
                b = torch.tensor(3)
                a[3] = 1
                return b
            elif df["Phase"][idx * 25 + 2] == "GallbladderRetraction":
                b = torch.tensor(6)
                a[6] = 1
                return b
            elif df["Phase"][idx * 25 + 2] == "CleaningCoagulation":
                b = torch.tensor(5)
                a[5] = 1
                return b
            elif df["Phase"][idx * 25 + 2] == "GallbladderPackaging":
                b = torch.tensor(4)
                a[4] = 1
                return b

    def get_labels(self):
        for name in self.video_names:
            with open(os.path.join(self.triplet_dir, name + '.txt')) as f:
                self.labels[name] = []
                for line in f.readlines():
                    li = list(map(int, line.split(',')))
                    frame_id = li[0]
                    triplet_id = li[1:]
                    self.labels[name].append([frame_id, triplet_id])

    def get_idx(self, idx):
        video_idx = 0
        for n in self.n_frames:
            if idx >= n:
                idx -= n
                video_idx += 1
            else:
                return idx, self.video_names[video_idx], '{0:06d}.png'.format(idx)
        raise ValueError('Index out of range.')