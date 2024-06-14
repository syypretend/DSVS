import copy
import os
import pandas as pd
import torch
from cvxpy.atoms.affine.wraps import psd_wrap
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
import math
import cvxpy as cp
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.utils.tensorboard import SummaryWriter
import time
from experiment.ex_test_multi_objective_binary_programming import ingredient, parse_config
from test import evaluate, evaluate_shot_level, evaluate_triplet
from tool.utils import AverageMeter

ex = Experiment('test_multi_objective_binary_programming', save_git_info=False, ingredients=[ingredient])
ex.observers.append(FileStorageObserver('log/sacred/test_multi_objective_binary_programming'))
local_time = time.localtime()
time_str = f'{local_time.tm_year}-{local_time.tm_mon:02}-{local_time.tm_mday:02}--' \
           f'{local_time.tm_hour:02}-{local_time.tm_min:02}-{local_time.tm_sec:02}'
log_dir = f'log/tensorboard/test_multi_objective_binary_programming/{time_str}'
writer = SummaryWriter(log_dir=log_dir)
print(f'use `$ tensorboard --logdir={log_dir} --bind_all` to open tensorboard')


def cos_similarity(vec1, vec2):
    cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos


def cosine_Matrix(_matrixA, _matrixB):
    _matrixA_matrixB = np.dot(_matrixA, _matrixB.transpose())
    _matrixA_norm = np.sqrt(np.multiply(_matrixA, _matrixA).sum(axis=1)).reshape(-1, 1)
    _matrixB_norm = np.sqrt(np.multiply(_matrixB, _matrixB).sum(axis=1)).reshape(-1, 1)
    return np.divide(_matrixA_matrixB, np.dot(_matrixA_norm, _matrixB_norm.transpose()))


def multi_objective_binary_programming(shot_importance, shot_similarity, shot_length, shot_boundary, min_size):
    each_phase_shot_importance_scores = []
    each_phase_shot_length = []
    each_shot_length = []
    each_pahse_shot_index = []
    shot_index = np.arange(len(shot_importance))
    idx = 0
    for i in range(len(shot_boundary)):
        each_shot_length.append(shot_boundary[i] + 1 - idx)
        idx = shot_boundary[i] + 1
    start = 0
    for i in shot_length:
        each_phase_shot_importance_scores.append(shot_importance[start:i + start])
        each_phase_shot_length.append(each_shot_length[start:i + start])
        each_pahse_shot_index.append(shot_index[start:i + start])
        start += i
    X = cp.Variable(len(shot_importance), integer=True)
    con = [0 <= X, X <= 1]
    A = np.zeros((len(each_phase_shot_length), len(shot_importance)), dtype=int)
    for i in range(len(each_phase_shot_importance_scores)):
        for j in range(len(each_pahse_shot_index[i])):
            if each_phase_shot_length[i][j] < min_size:
                con.append(X[each_pahse_shot_index[i][j]] == 0)
            else:
                A[i][each_pahse_shot_index[i][j]] = 1
    for i in range(len(A)):
        if 1 not in A[i]:
            continue
        con.append(A[i] @ X <= math.ceil((len(each_phase_shot_length[i]) ** 2) / len(each_shot_length)))
        con.append(A[i] @ X >= 1)
    is_strict = True
    alpha = 0.3
    for i in range(100):
        con_standby = copy.copy(con)
        if is_strict:
            con.append(each_shot_length @ X <= alpha * sum(each_shot_length))
        obj_importance_scores = cp.Minimize(-1 * np.array(shot_importance) @ X/len(shot_importance))
        prob_importance_scores = cp.Problem(obj_importance_scores, con)
        prob_importance_scores.solve(solver='GLPK_MI')
        value_i = prob_importance_scores.value
        print(f"ideal_objective_importance:{-1 * value_i}")
        obj_similarity = cp.Minimize(cp.norm(psd_wrap(shot_similarity) @ X, 2)/len(shot_importance))
        prob_similarity = cp.Problem(obj_similarity, con)
        prob_similarity.solve(solver="CPLEX")
        value_f = prob_similarity.value
        print(f"ideal_objective_similarity:{value_f}")
        if math.isinf(float(value_i)) or math.isinf(float(value_f)) or value_f is None:
            con = con_standby
            alpha += 0.01
        else:
            break
    obj = cp.Minimize(
         (-1 * np.array(shot_importance) @ X/len(shot_importance) - value_i) + (cp.norm(psd_wrap(shot_similarity) @ X, 2)/len(shot_importance) - value_f))
    prob = cp.Problem(obj, con)
    prob.solve(solver='CPLEX')
    print(
        f"importance_scores: {(np.array(shot_importance) @ X/len(shot_importance)).value} similarity: {(cp.norm(psd_wrap(shot_similarity) @ X, 2)/len(shot_importance)).value}")
    print(f"length_ratio: {sum(X.value * np.array(each_shot_length).transpose()) / sum(each_shot_length)}")
    return X.value


@ex.main
def test_multi_objective_binary_programming(_config):
    config = parse_config(_config)
    a = np.load(config.hidden_data_dir, allow_pickle=True)
    hidden_data = a.item()
    min_size = config.min_size
    jump = config.jump
    informative = AverageMeter()
    diversity = AverageMeter()
    summary = {}
    for video_name in hidden_data.keys():
        summary[video_name] = {}
        each_phase_instrument = hidden_data[video_name]["each_phase_instrument"]
        each_phase_frame_id = hidden_data[video_name]["each_phase_frame_id"]
        each_phase_importance_scores = hidden_data[video_name]["each_phase_importance_scores"]
        each_phase_feature = hidden_data[video_name]["each_phase_feature"]
        a = 0
        for length in each_phase_frame_id:
            a+=len(length)
        index = min(a, len(hidden_data[video_name]["semantic_label"]))
        semantic_label = hidden_data[video_name]["semantic_label"][:index]
        shots = []
        shot_length = []
        shot_boundary = []
        bkps = []
        for i in range(len(each_phase_instrument)):
            f = each_phase_instrument[i]
            if f.shape[0] < min_size or f.shape[0] < jump:
                change_point = [len(f)]
            else:
                algo = rpt.Pelt(model="l2", jump=jump, min_size=min_size).fit(np.mean(np.array(f), axis=1))
                change_point = algo.predict(pen=config.pen)  # return the index of shot
            shots.append(change_point)
            shot_length.append(len(change_point))
            for j in change_point:
                shot_boundary.append(each_phase_frame_id[i][j - 1])
            bkps.append(each_phase_frame_id[i][-1])
        fig, ax_arr = rpt.display(np.mean(np.concatenate(each_phase_instrument, axis=0), axis=1), bkps, shot_boundary,
                                  figsize=(20, 6))
        print(f"video_name: {video_name}; num_shot: {sum(shot_length)}; shot: {shot_boundary}")
        each_shot_importance_scores = []
        each_shot_feature = []
        each_shot_semantic_label = []
        importance_scores_display = []
        each_shot_frame_id = []
        importance_scores = np.concatenate(each_phase_importance_scores, axis=0)
        feature = np.concatenate(each_phase_feature, axis=0)
        frame_id = np.concatenate(each_phase_frame_id, axis=0)
        start_idx = 0
        for i in range(len(shot_boundary)):
            length = shot_boundary[i] + 1 - start_idx
            each_shot_importance_scores.append(importance_scores[start_idx:shot_boundary[i] + 1].mean())
            each_shot_feature.append(feature[start_idx:shot_boundary[i] + 1])
            each_shot_semantic_label.append(semantic_label[start_idx:shot_boundary[i] + 1])
            each_shot_frame_id.append(frame_id[start_idx:shot_boundary[i] + 1])
            for j in range(length):
                importance_scores_display.append(each_shot_importance_scores[i])
            start_idx = shot_boundary[i] + 1
        similarity_matrix = np.zeros((len(shot_boundary), len(shot_boundary)))
        restart = True
        decimal_point = 2
        while restart:
            for i in range(len(each_shot_feature)):
                for j in range(len(each_shot_feature)):
                    if i == j:
                        similarity_matrix[i, j] = 1
                    else:
                        cos_sim = cosine_Matrix(each_shot_semantic_label[i], each_shot_semantic_label[j])
                        similarity = cos_sim.sum() / (cos_sim.shape[0] * cos_sim.shape[1])
                        similarity_matrix[i, j] = round(similarity, decimal_point)
                        similarity_matrix[j, i] = similarity_matrix[i, j]
            try:
                result = multi_objective_binary_programming(each_shot_importance_scores, np.multiply(similarity_matrix, similarity_matrix), shot_length,shot_boundary, min_size)
                restart = False
            except:
                decimal_point += 1
        plt.plot(importance_scores_display, color="red")
        writer.add_figure(f"shot_boundary/{video_name}", fig, global_step=None)
        print(result)
        selected_frame_id = []
        for i in range(len(result)):
            if result[i] == 1:
                for j in each_shot_frame_id[i]:
                    selected_frame_id.append(j)

        triplet_dir = "/home/syy/old/cholecT50_train/triplet"
        triplet_label = []
        with open(os.path.join(triplet_dir, video_name + ".txt"), 'r') as triplet:
            for line in triplet.readlines():
                li = list(map(int, line.split(',')))
                # frame_id = li[0]
                triplet_id = li[1:]
                triplet_label.append(triplet_id)
        triplet_label = np.array(triplet_label)
        diversity_matrix = cosine_Matrix(triplet_label[selected_frame_id], triplet_label[selected_frame_id])
        for i in range(len(selected_frame_id)):
            for j in range(len(selected_frame_id)):
                if 1 not in triplet_label[selected_frame_id[i]] and 1 not in triplet_label[selected_frame_id[j]]:
                    diversity_matrix[i, j] = 1
                    diversity_matrix[j, i] = 1
                elif 1 in triplet_label[selected_frame_id[i]] and 1 in triplet_label[selected_frame_id[j]]:
                    continue
                else:
                    diversity_matrix[i, j] = 0
                    diversity_matrix[j, i] = 0

        summary[video_name]["keyframe"] = selected_frame_id
        summary[video_name]["origin_video"] = list(frame_id)
        diversity_score = (1 - diversity_matrix).sum() / (len(selected_frame_id) * (len(selected_frame_id) - 1))
        informativeness, _, _ = evaluate_triplet(video_name, selected_frame_id, list(frame_id))
        informative.update(informativeness)
        diversity.update(diversity_score)
        print(f"video: {video_name}, informativeness: {informativeness:.5f}, diversity_score: {diversity_score}")
        print("============================================================================")
    print(f"informative: {informative.avg} diversity: {diversity.avg}")

if __name__ == '__main__':
    ex.run_commandline()


