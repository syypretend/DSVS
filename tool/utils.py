import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from scipy.stats import entropy

def weight_ivt(i, v, t):
    """
    :param i: array,shape(num_frames, 6)
    :param v: array,shape(num_frames, 10)
    :param t: array,shape(num_frames, 15)
    :return:  array
    """
    weight_i = np.zeros((i.shape[0], 1))
    weight_v = np.zeros((v.shape[0], 1))
    weight_t = np.zeros((t.shape[0], 1))
    for j in range(i.shape[0]):
        if j + 1 < i.shape[0]:
            weight_i[j] = np.linalg.norm(i[j]-i[j + 1])
            weight_v[j] = np.linalg.norm(v[j]-v[j + 1])
            weight_t[j] = np.linalg.norm(t[j]-t[j + 1])
    return weight_i, weight_v, weight_t


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


def _compute_AP(gt_labels, pd_probs, valid=None):
    gt_instances = np.sum(gt_labels, axis=0)
    pd_instances = np.sum(pd_probs, axis=0)
    computed_ap = average_precision_score(gt_labels, pd_probs, average=None)
    actual_ap = []
    num_classes = np.shape(gt_labels)[-1]
    for k in range(num_classes):
        if ((gt_instances[k] != 0) or (pd_instances[k] != 0)) and not np.isnan(computed_ap[k]):
            actual_ap.append(computed_ap[k])
        else:
            actual_ap.append("n/a")
    return actual_ap


def average_by_videos(results):
    """ Compute the average AP of each triplet class across all the videos
        and mean AP of the model on the triplet predictions.
        Args:
            results:   1D (batch of) vector of AP for each class. One member of the batch corresponds
                       to one video
        Returns:
            AP:   1D vector[N] of AP for each class averaged by videos
    """
    n = results.shape[-1]
    AP = []
    for j in range(n):
        x = results[:, j]
        x = np.mean([float(a) for a in x if (str(a) != 'n/a')])
        if np.isnan(x):
            AP.append("n/a")
        else:
            AP.append(x)
    mAP = np.mean([i for i in AP if i != 'n/a'])
    return np.array(AP), mAP


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_metrics(metrics: dict, epoch, sacred_ex=None, tb_writer=None):
    parsed = {}

    def r_parse(metrics: dict, key_pref=''):
        for k, v in metrics.items():
            full_key = '/'.join([key_pref, k]) if key_pref != '' else k
            if isinstance(v, dict):
                r_parse(v, full_key)  # `v` is a dict
            else:
                parsed[full_key] = v  # `v` is a value

    r_parse(metrics)
    for k, v in parsed.items():
        if sacred_ex is not None:
            sacred_ex.log_scalar(k, v, epoch)
        if tb_writer is not None:
            tb_writer.add_scalar(k, v, epoch)


# suggest update
def calc_weight_loss(dataloader, maps_dict, i_class=6, v_class=10, t_class=15):
    i_static = np.zeros(i_class)
    v_static = np.zeros(v_class)
    t_static = np.zeros(t_class)

    for _, _, _, y in dataloader:
        for idx in np.where(y == 1)[1]:
            i_idx = maps_dict[idx, 1]
            v_idx = maps_dict[idx, 2]
            t_idx = maps_dict[idx, 3]
            i_static[i_idx] += 1
            v_static[v_idx] += 1
            t_static[t_idx] += 1

    i_static[i_static == 0] = 1
    v_static[v_static == 0] = 1
    t_static[t_static == 0] = 1

    i_median = np.median(i_static)
    v_median = np.median(v_static)
    t_median = np.median(t_static)

    i_weight = i_median / i_static
    v_weight = v_median / v_static
    t_weight = t_median / t_static

    weight = np.concatenate([i_weight, v_weight, t_weight])

    return torch.from_numpy(weight)


def cal_adj_matrix():
    a = np.zeros((7, 7), dtype=int)
    for txt_name in os.listdir(r"E:\CholecT50-train\phase"):
        with open(os.path.join(r"E:\CholecT50-train\phase", txt_name), "r") as f:
            phase_list = []
            for line in f.readlines()[1:]:
                li = line.split()
                phase = li[1]
                if phase == "Preparation":
                    phase_list.append(0)
                elif phase == "CalotTriangleDissection":
                    phase_list.append(1)
                elif phase == "ClippingCutting":
                    phase_list.append(2)
                elif phase == "GallbladderDissection":
                    phase_list.append(3)
                elif phase == "GallbladderPackaging":
                    phase_list.append(4)
                elif phase == "CleaningCoagulation":
                    phase_list.append(5)
                elif phase == "GallbladderRetraction":
                    phase_list.append(6)
                else:
                    continue
            for i in range(len(phase_list) - 1):
                a[phase_list[i], phase_list[i + 1]] = 1
                # if a[3, 5] == 1:
                #     print(txt_name)
    return a

def evaluate_triplet(video_name, keyframe: list, origin_video: list):
    triplet_dir = r"E:\CholecT50-train\triplet"
    triplet_label = []
    with open(os.path.join(triplet_dir, video_name + ".txt"), 'r') as triplet:
        for line in triplet.readlines():
            li = list(map(int, line.split(',')))
            frame_id = li[0]
            triplet_id = li[1:]
            triplet_label.append(triplet_id)
    keyframe_triplet = np.zeros(100)
    origin_triplet = np.zeros(100)
    for i in keyframe:
        keyframe_triplet += np.array(triplet_label)[i]
    for i in origin_video:
        origin_triplet += np.array(triplet_label)[i]
    b = 0
    c = 0
    for i in keyframe_triplet:
        if i != 0:
            b += 1
    for i in origin_triplet:
        if i != 0:
            c += 1
    completeness = b/c
    prob = np.array(triplet_label).sum(axis=0) / len(triplet_label)
    group = {}
    for frame_id in np.array(keyframe):
        group[frame_id] = []
    log_prob = []
    for i in range(len(prob)):
        if prob[i] == 0:
            log_prob.append(0)
        else:
            log_prob.append(-1 * np.log2(prob[i]))
    self_information_key = []
    for i in keyframe:
        if len(np.where(np.array(triplet_label)[i] == 1)[0]) == 0:
            self_information = 0
            self_information_key.append(self_information)
        else:
            self_information = 0
            for j in np.where(np.array(triplet_label)[i] == 1)[0]:
                self_information += log_prob[j]
            self_information_key.append(self_information)
    self_information_origin = []
    for i in origin_video:
        if len(np.where(np.array(triplet_label)[i] == 1)[0]) == 0:
            self_information = 0
            self_information_origin.append(self_information)
        else:
            self_information = 0
            for j in np.where(np.array(triplet_label)[i] == 1)[0]:
                self_information += log_prob[j]
            self_information_origin.append(self_information)
        a = [abs(self_information - information) for information in self_information_key]
        idx = a.index(min(a))
        if len([*filter(lambda x: x >= max(self_information_key), a)]) > 0:
            group[keyframe[idx]].append(self_information_key[idx])
        else:
            group[keyframe[idx]].append(self_information)
    compactness_key = []
    compactness_origin = []
    for i in range(len(self_information_key) - 1):
        compactness_key.append(abs(self_information_key[i] - self_information_key[i + 1]))
    for i in range(len(self_information_origin) - 1):
        compactness_origin.append(abs(self_information_origin[i] - self_information_origin[i + 1]))
    compactness_ratio = np.array(compactness_key).mean() / np.array(compactness_origin).mean()

    variance_group = []
    for key in group.keys():
        if not len(group[key]):
            variance_group.append(0)
            # continue
        else:
            variance_group.append(np.array(
                [((self_information_key[np.where(np.array(keyframe) == key)[0].item()] - information) ** 2) for
                 information in group[key]]).mean())

    representativeness_ratio = np.array(variance_group).mean()
    return compactness_ratio, representativeness_ratio, completeness

def information_contribution_JS_two(i_prob, v_prob, t_prob, comp="IVT"):
    indexes = []
    with open("E:\\CholecT50-train\\dict\\maps.txt") as f:
        for line in f:
            values = line.split(',')
            if '#' in values[0]:
                continue
            trip = list(map(int, values[1:4]))

            if comp == "IVT":
                indexes.append(trip)
            elif comp == "IV":
                indexes.append(trip[:2])
            elif comp == "IT":
                indexes.append([trip[0], trip[2]])
            elif comp == "VT":
                indexes.append([trip[1], trip[2]])
            elif comp == "I":
                indexes.append([trip[0]])
            elif comp == "V":
                indexes.append([trip[1]])
            elif comp == "T":
                indexes.append([trip[2]])

        indexes = np.array(indexes)
    if comp == "IVT":
        valid_triplets = torch.zeros((len(i_prob), i_prob.shape[1], v_prob.shape[1], t_prob.shape[1]))
        for i in range(i_prob.shape[0]):
            for j in indexes:
                valid_triplets[i][j[0]][j[1]][j[2]] = (i_prob[i][j[0]] + v_prob[i][j[1]] + t_prob[i][j[2]]) / 3
    elif comp == "IV":
        valid_triplets = torch.zeros((len(i_prob), i_prob.shape[1], v_prob.shape[1]))
        for i in range(i_prob.shape[0]):
            for j in indexes:
                valid_triplets[i][j[0]][j[1]] = (i_prob[i][j[0]] + v_prob[i][j[1]]) / 2
    elif comp == "IT":
        valid_triplets = torch.zeros((len(i_prob), i_prob.shape[1], t_prob.shape[1]))
        for i in range(i_prob.shape[0]):
            for j in indexes:
                valid_triplets[i][j[0]][j[1]] = (i_prob[i][j[0]] + t_prob[i][j[1]]) / 2
    elif comp == "VT":
        valid_triplets = torch.zeros((len(i_prob), v_prob.shape[1], t_prob.shape[1]))
        for i in range(i_prob.shape[0]):
            for j in indexes:
                valid_triplets[i][j[0]][j[1]] = (v_prob[i][j[0]] + t_prob[i][j[1]]) / 2
    elif comp == "I":
        valid_triplets = torch.zeros((len(i_prob), i_prob.shape[1]))
        for i in range(i_prob.shape[0]):
            for j in indexes:
                valid_triplets[i][j[0]] = i_prob[i][j[0]] / 1
    elif comp == "V":
        valid_triplets = torch.zeros((len(i_prob), v_prob.shape[1]))
        for i in range(i_prob.shape[0]):
            for j in indexes:
                valid_triplets[i][j[0]] = v_prob[i][j[0]] / 1
    elif comp == "T":
        valid_triplets = torch.zeros((len(i_prob), t_prob.shape[1]))
        for i in range(i_prob.shape[0]):
            for j in indexes:
                valid_triplets[i][j[0]] = t_prob[i][j[0]] / 1
    weight = np.zeros((i_prob.shape[0], 1))
    for j in range(i_prob.shape[0]):
        if j + 1 < i_prob.shape[0]:
            distance = []
            for i in range(len(valid_triplets.numpy()[j].flatten())):
                if valid_triplets.numpy()[j].flatten()[i] == 0:
                    continue
                else:
                    a = valid_triplets.numpy()[j].flatten()[i]
                    b = valid_triplets.numpy()[j + 1].flatten()[i]
                    m = 0.5 * (a + b)
                    js = 0.5 * (entropy([a, 1 - a], [m, 1 - m]) + entropy([b, 1 - b], [m, 1 - m]))
                    distance.append(js)
            weight[j] = np.sum(distance)
    return weight