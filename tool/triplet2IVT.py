# 由triplet标签获得I,V,T的标签
import numpy as np
import os
import torch


def triplet2IVT_sequence(map_path, trip_label):
    maps_dict = np.genfromtxt(map_path, dtype=int, comments='#', delimiter=',', skip_header=0)
    indices = np.where(trip_label == 1)
    i_label = torch.zeros((trip_label.shape[0], trip_label.shape[1], 6), dtype=torch.float)
    v_label = torch.zeros((trip_label.shape[0], trip_label.shape[1], 10), dtype=torch.float)
    t_label = torch.zeros((trip_label.shape[0], trip_label.shape[1], 15), dtype=torch.float)
    i_idx = maps_dict[indices[2], 1]
    v_idx = maps_dict[indices[2], 2]
    t_idx = maps_dict[indices[2], 3]
    i_label[indices[0], indices[1], i_idx] = 1
    v_label[indices[0], indices[1], v_idx] = 1
    t_label[indices[0], indices[1], t_idx] = 1
    return i_label, v_label, t_label


def triplet2IVT(map_path, trip_label):
    maps_dict = np.genfromtxt(map_path, dtype=int, comments='#', delimiter=',', skip_header=0)
    indices = np.where(trip_label == 1)
    i_label = torch.zeros((trip_label.shape[0], 6), dtype=torch.float)
    v_label = torch.zeros((trip_label.shape[0], 10), dtype=torch.float)
    t_label = torch.zeros((trip_label.shape[0], 15), dtype=torch.float)
    i_idx = maps_dict[indices[1], 1]
    v_idx = maps_dict[indices[1], 2]
    t_idx = maps_dict[indices[1], 3]
    i_label[indices[0], i_idx] = 1
    v_label[indices[0], v_idx] = 1
    t_label[indices[0], t_idx] = 1
    return i_label, v_label, t_label