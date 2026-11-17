from torch.utils.data import Sampler
import os
import numpy as np


def get_train_idx(config):
    # each video length
    each_video_length = []
    if not config.DEBUG:
        for name in [config.video_names[i] for i in config.train_index]:
            frames = len(os.listdir(os.path.join(config.data_dir, name)))
            each_video_length.append(frames)
    else:
        for name in [config.video_names[i] for i in config.train_index[:2]]:
            frames = len(os.listdir(os.path.join(config.data_dir, name)))
            each_video_length.append(frames)
    count = 0
    # start index of each sequence
    idx = []
    for i in range(len(each_video_length)):
        for j in range(count, count + (each_video_length[i] + 1 - config.sequence_length)):
            idx.append(j)
        count += each_video_length[i]
    # index of each sequence
    np.random.shuffle(idx)
    train_idx = []
    for i in range(len(idx)):
        for j in range(config.sequence_length):
            train_idx.append(idx[i] + j)
    return [idx, train_idx, each_video_length]


def get_valid_idx(config):
    # each video length
    each_video_length = []
    if not config.DEBUG:
        for name in [config.video_names[i] for i in config.valid_index]:
            frames = len(os.listdir(os.path.join(config.data_dir, name)))
            each_video_length.append(frames)
    else:
        for name in [config.video_names[i] for i in config.valid_index[:2]]:
            frames = len(os.listdir(os.path.join(config.data_dir, name)))
            each_video_length.append(frames)
    count = 0
    # start index of each sequence
    idx = []
    for i in range(len(each_video_length)):
        for j in range(count, count + (each_video_length[i] + 1 - config.sequence_length)):
            idx.append(j)
        count += each_video_length[i]
    # index of each sequence
    valid_idx = []
    for i in range(len(idx)):
        for j in range(config.sequence_length):
            valid_idx.append(idx[i] + j)
    return [idx, valid_idx, each_video_length]


def get_test_idx(config):
    # each video length
    each_video_length = []
    if not config.DEBUG:
        for name in [config.video_names[i] for i in config.test_index]:
            frames = len(os.listdir(os.path.join(config.data_dir, name)))
            each_video_length.append(frames)
    else:
        for name in [config.video_names[i] for i in config.test_index[:2]]:
            frames = len(os.listdir(os.path.join(config.data_dir, name)))
            each_video_length.append(frames)
    count = 0
    # start index of each sequence
    idx = []
    for i in range(len(each_video_length)):
        for j in range(count, count + (each_video_length[i] + 1 - config.sequence_length)):
            idx.append(j)
        count += each_video_length[i]
    # index of each sequence
    test_idx = []
    for i in range(len(idx)):
        for j in range(config.sequence_length):
            test_idx.append(idx[i] + j)
    return [idx, test_idx, each_video_length]


# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
