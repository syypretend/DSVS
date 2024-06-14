import math
import matplotlib.pyplot as plt
from information_contribution import information_contribution_JS_two
from ivtmetrics.recognition import Recognition
import numpy as np
from test import evaluate
from tool.PKI_calibration import gridsearchcv_video, PKI
from tool.sequence_preprocess import get_test_idx
from tool.triplet2IVT import triplet2IVT
from tool.utils import AverageMeter, cal_adj_matrix, weight_ivt
import pandas as pd
import ruptures as rpt
from tool.utils import sigmoid
import cvxpy as cp
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from sklearn import metrics
from dataset import CholecVideo
from experiment.ex_test_MRTnet import ingredient, parse_config
from model.MRTnet import MRTnet, MRTnet_trans, MRTnet_lstm, MRTnet_single_lstm
# sacred
from tool.sequence_preprocess import SeqSampler, get_valid_idx
from tool.utils import calc_weight_loss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
ex = Experiment('test_MRTnet', save_git_info=False, ingredients=[ingredient])
ex.observers.append(FileStorageObserver('log/sacred/test_MRTnet'))

# tensorboard
from torch.utils.tensorboard import SummaryWriter
import time

local_time = time.localtime()
time_str = f'{local_time.tm_year}-{local_time.tm_mon:02}-{local_time.tm_mday:02}--' \
           f'{local_time.tm_hour:02}-{local_time.tm_min:02}-{local_time.tm_sec:02}'
log_dir = f'log/tensorboard/test_MRTnet/{time_str}'
writer = SummaryWriter(log_dir)
print(f'use `$ tensorboard --logdir={log_dir} --bind_all` to open tensorboard')


@ex.main
def test_MRTnet_trans(_config):
    config = parse_config(_config)
    maps_dict = np.genfromtxt(config.map_path, dtype=int, comments='#', delimiter=',', skip_header=0)
    test_transform = transforms.Compose([
        transforms.Resize((256, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    torch.backends.cudnn.benchmark = True

    test_dataset = CholecVideo(
        data_dir=config.data_dir,
        triplet_dir=config.triplet_dir,
        phase_dir=config.phase_dir,
        sequence_length=config.sequence_length,
        video_names=[config.video_names[i] for i in config.valid_index],
        transform=test_transform,
        batch_size=config.batch_size
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True,
                             num_workers=config.num_workers,
                             sampler=SeqSampler(test_dataset, get_valid_idx(config)[1]),
                             drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MRTnet_lstm(backbone="resnet18", grad_norm=config.grad_norm, use_crf=True)
    state = torch.load(str(config.checkpoint), map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    print(f'Load model state dict from {config.checkpoint}, epoch: {state["epoch"]}, mAP_i: {state["mAP_i"]}, '
          f'mAP_v: {state["mAP_v"]}, mAP_t: {state["mAP_t"]}, acc_p: {state["acc_p"]}')
    model = nn.DataParallel(model).to(device)
    print("CRF transitions matrix:")
    print(model.module.crf.transitions.data)
    recognize_i = Recognition(num_class=6)
    recognize_v = Recognition(num_class=10)
    recognize_t = Recognition(num_class=15)
    acc_p_meter = AverageMeter()
    recognize_i.reset()
    recognize_v.reset()
    recognize_t.reset()
    p_pre = []
    i_pre = []
    v_pre = []
    t_pre = []
    i_tar = []
    v_tar = []
    t_tar = []
    p_tar = []
    cat_features = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (frame_idx, v_name, input, target, target_p) in enumerate(test_loader):
            input = input.to(device)
            target_p = target_p.to(device)
            ivt_label = triplet2IVT(config.map_path, target)
            i_label, v_label, t_label = ivt_label[0].to(device), ivt_label[1].to(device), ivt_label[2].to(device)
            # ---- valid model ---- #
            pred_i, pred_v, pred_t, pred_p, feature, _ = model(input, config.sequence_length, target_p)
            
            pred_i = pred_i.view(-1, 6)
            pred_v = pred_v.view(-1, 10)
            pred_t = pred_t.view(-1, 15)
            feature = feature.contiguous().view(config.batch_size, -1)
            pred_p = pred_p.contiguous().view(-1, 7)[config.sequence_length - 1::config.sequence_length]
            target_p = target_p[config.sequence_length - 1::config.sequence_length].to(device)

            # ---- valid mAP ---- #
            # mAP_i, mAP_v, mAP_t
            recognize_i.update(i_label.detach().cpu().numpy(), pred_i.detach().cpu().numpy())
            recognize_v.update(v_label.detach().cpu().numpy(), pred_v.detach().cpu().numpy())
            recognize_t.update(t_label.detach().cpu().numpy(), pred_t.detach().cpu().numpy())

            # collect all output
            i_pre.append(pred_i.detach().cpu().numpy())
            v_pre.append(pred_v.detach().cpu().numpy())
            t_pre.append(pred_t.detach().cpu().numpy())
            p_pre.append(pred_p.detach().cpu().numpy())
            i_tar.append(i_label.detach().cpu().numpy())
            v_tar.append(v_label.detach().cpu().numpy())
            t_tar.append(t_label.detach().cpu().numpy())
            p_tar.append(target_p.detach().cpu().numpy())
            cat_features.append(feature.detach().cpu().numpy())
    # compute mAP with sequence_length
    result_i = recognize_i.compute_AP("i")
    result_v = recognize_v.compute_AP("v")
    result_t = recognize_t.compute_AP('t')
    print(
        f'[[test] [mAP_i: {result_i["mAP"]:.5f}, mAP_v: {result_v["mAP"]:.5f}, mAP_t: {result_t["mAP"]:.5f}]')

    # compute mAP by each video without sequence_length
    i_preds = np.concatenate(i_pre, axis=0)
    v_preds = np.concatenate(v_pre, axis=0)
    t_preds = np.concatenate(t_pre, axis=0)
    p_preds = np.concatenate(p_pre, axis=0)
    i_target = np.concatenate(i_tar, axis=0)
    v_target = np.concatenate(v_tar, axis=0)
    t_target = np.concatenate(t_tar, axis=0)
    p_target = np.concatenate(p_tar, axis=0)
    features = np.concatenate(cat_features, axis=0)
    each_video_length = get_valid_idx(config)[2]
    count = 0
    recognize_i.reset_global()
    recognize_v.reset_global()
    recognize_t.reset_global()
    # each video
    compactness = []
    representativeness = []
    hidden_data = {}
    a = []
    b = []
    c = []
    trans_SVnet_a = []
    trans_SVnet_b = []
    trans_SVnet_c = []
    TMRnet_a = []
    TMRnet_b = []
    TMRnet_c = []
    labels_phase = []
    preds_phase = []
    for i in range(len(each_video_length)):
        video_i_prediction = []
        video_v_prediction = []
        video_t_prediction = []
        video_p_prediction = []
        video_i_target = []
        video_v_target = []
        video_t_target = []
        video_p_target = []
        video_feature = []
        if not config.DEBUG:
            video_name = [config.video_names[j] for j in config.valid_index][i]
        else:
            video_name = [config.video_names[j] for j in config.valid_index[:2]][i]
        # drop last  problem process
        if i == len(each_video_length) - 1:
            video_length = int((len(test_loader) * config.batch_size - count) / config.sequence_length)
            # ---- prediction without by sigmoid ---- #
            video_i_prediction.append(i_preds[count:count + (
                    video_length + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_v_prediction.append(v_preds[count:count + (
                    video_length + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_t_prediction.append(t_preds[count:count + (
                    video_length + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_p_prediction.append(p_preds[
                                      int(count / config.sequence_length):int(count / config.sequence_length) + (
                                              video_length + 1 - config.sequence_length)])

            # ---- target ---- #
            video_i_target.append(i_target[count:count + (
                    video_length + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_v_target.append(v_target[count:count + (
                    video_length + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_t_target.append(t_target[count:count + (
                    video_length + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_feature.append(features[count:count + (
                    video_length + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_p_target.append(p_target[
                                      int(count / config.sequence_length):int(count / config.sequence_length) + (
                                              video_length + 1 - config.sequence_length)])
            count += (video_length + 1 - config.sequence_length) * config.sequence_length
        else:
            # ---- prediction without by sigmod ---- #
            video_i_prediction.append(i_preds[count:count + (each_video_length[
                                                                 i] + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_v_prediction.append(v_preds[count:count + (each_video_length[
                                                                 i] + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_t_prediction.append(t_preds[count:count + (each_video_length[
                                                                 i] + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_p_prediction.append(p_preds[
                                      int(count / config.sequence_length):int(count / config.sequence_length) + (
                                              each_video_length[i] + 1 - config.sequence_length)])

            # ---- target ---- #
            video_i_target.append(i_target[count:count + (each_video_length[
                                                              i] + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_v_target.append(v_target[count:count + (each_video_length[
                                                              i] + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_t_target.append(t_target[count:count + (each_video_length[
                                                              i] + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_feature.append(features[count:count + (each_video_length[
                                                             i] + 1 - config.sequence_length) * config.sequence_length:config.sequence_length])
            video_p_target.append(p_target[
                                      int(count / config.sequence_length):int(count / config.sequence_length) + (
                                              each_video_length[i] + 1 - config.sequence_length)])
            count += (each_video_length[i] + 1 - config.sequence_length) * config.sequence_length

        i_preds_np = np.concatenate(video_i_prediction, axis=0)
        v_preds_np = np.concatenate(video_v_prediction, axis=0)
        t_preds_np = np.concatenate(video_t_prediction, axis=0)
        p_preds_np = np.concatenate(video_p_prediction, axis=0)  # [frames, 7]
        i_target_np = np.concatenate(video_i_target, axis=0)
        v_target_np = np.concatenate(video_v_target, axis=0)
        t_target_np = np.concatenate(video_t_target, axis=0)
        p_target_np = np.concatenate(video_p_target, axis=0)
        features_np = np.concatenate(video_feature, axis=0)  # [frames, dimensions]
        min_len = min(len(i_target_np),len(i_preds_np))
        recognize_i.update(i_target_np[:min_len,:], i_preds_np[:min_len,:])
        recognize_i.video_end()
        recognize_v.update(v_target_np[:min_len,:], v_preds_np[:min_len,:])
        recognize_v.video_end()
        recognize_t.update(t_target_np[:min_len,:], t_preds_np[:min_len,:])
        recognize_t.video_end()

        # acc_p
        pred_labels = model.module.crf.decode(torch.from_numpy(p_preds_np).view(1, -1, 7).cuda())
        pred_phase = torch.tensor(pred_labels).detach().cpu().view(-1)
        if config.adj_matrix is None:
            adj_matrix = cal_adj_matrix()
            np.save('data/adj_matrix.npy', adj_matrix)
            ex.add_artifact('data/adj_matrix.npy')
        else:
            ex.add_resource(config.adj_matrix)
            adj_matrix = np.load(config.adj_matrix)
        a.append(pred_phase.numpy())
        b.append(p_preds_np)
        c.append(p_target_np)


        # is_test = False to get alpha and beta; is_test = True to test video summary compactness by using alpha and beta
        pred_phase = pred_phase.numpy()
        if config.is_test:
            pred_phase = PKI(adj_matrix, pred_phase, p_preds_np, p_target_np, config.alpha, config.beta)
        video_corrects_phase = torch.sum(torch.from_numpy(pred_phase) == torch.from_numpy(p_target_np).view(-1))
        labels_phase.append(p_target_np.reshape(-1, 1))
        preds_phase.append(pred_phase.reshape(-1, 1))
        acc_p_meter.update(video_corrects_phase/len(p_target_np))

        # visual phase
        PHASES = [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingAndCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningAndCoagulation",
            "GallbladderRetraction"
        ]
        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(111)
        ax.set_yticks([], [])
        ax.set_title(f"{video_name}")
        im = ax.pcolormesh(pred_phase.reshape(1, -1), cmap="Set2", vmin=0, vmax=7)
        cbar = fig.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], orientation="vertical", drawedges=True)
        cbar.ax.set_yticklabels(PHASES)
        writer.add_figure(f"pred_phase/{video_name}", fig, i)

        fig_t = plt.figure(figsize=(10, 2))
        ax = fig_t.add_subplot(111)
        ax.set_yticks([], [])
        ax.set_title(f"{video_name}")
        im_t = ax.pcolormesh(p_target_np.reshape(1, -1), cmap="Set2", vmin=0, vmax=7)
        cbar_t = fig_t.colorbar(im_t, ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], orientation="vertical", drawedges=True)
        cbar_t.ax.set_yticklabels(PHASES)
        writer.add_figure(f"target_phase/{video_name}", fig_t, i)

        # compute importance scores
        i_prob = sigmoid(i_preds_np)
        v_prob = sigmoid(v_preds_np)
        t_prob = sigmoid(t_preds_np)
        frames_importance_scores = information_contribution_JS_two(i_prob, v_prob, t_prob, comp="IVT")


        # record importance_scores
        for idx in range(len(frames_importance_scores)):
            writer.add_scalar(f"importance_scores/{video_name}", frames_importance_scores[idx], idx)

        # get index of every phase
        phase_start_idx = []
        for idx in range(len(pred_phase) - 1):
            if pred_phase[idx] != pred_phase[idx + 1]:
                phase_start_idx.append(idx + 1)
        phase_start_idx.append(len(pred_phase))
        phase = []
        for p in range(len(pred_phase)-1):
            if pred_phase[p] != pred_phase[p+1]:
                phase.append(pred_phase[p])
        phase.append(pred_phase[-1])
        start = 0
        frame_id = np.arange(len(pred_phase))
        each_phase_feature = []
        each_phase_instrument = []
        each_phase_frame_id = []
        each_phase_importance_scores = []
        for k in range(len(phase_start_idx)):
            each_phase_feature.append(features_np[start:phase_start_idx[k]])
            each_phase_instrument.append(sigmoid(i_preds_np)[start:phase_start_idx[k]])
            each_phase_frame_id.append(frame_id[start:phase_start_idx[k]])
            each_phase_importance_scores.append(frames_importance_scores[start:phase_start_idx[k]])
            start = phase_start_idx[k]
        hidden_data[video_name] = {"each_phase_instrument": each_phase_instrument, "each_phase_frame_id": each_phase_frame_id,
                                   "each_phase_feature": each_phase_feature, "each_phase_importance_scores": each_phase_importance_scores,
                                   "phase": phase, "semantic_label": np.concatenate((sigmoid(i_preds_np), sigmoid(v_preds_np), sigmoid(t_preds_np)), axis=1),
                                   "pred_phase": pred_phase}
    if config.is_test:
        print("test to generate hidden_data.npy !")
        # np.save("IC/hidden_data_JS.npy", hidden_data)
    else:
        print("valid to get alpha and beta !")
        correct, _ = gridsearchcv_video(adj_matrix, a, b, c) # ours
        print(f"correct: {correct}")
    labels_phase = list(np.vstack(labels_phase).reshape(-1))
    preds_phase = list(np.vstack(preds_phase).reshape(-1))
    recall_phase = metrics.recall_score(labels_phase, preds_phase, average='macro')
    precision_phase = metrics.precision_score(labels_phase, preds_phase, average='macro')
    jaccard_phase = metrics.jaccard_score(labels_phase, preds_phase, average='macro')
    precision_each_phase = metrics.precision_score(labels_phase, preds_phase, average=None)
    recall_each_phase = metrics.recall_score(labels_phase, preds_phase, average=None)
    jaccard_each_phase = metrics.jaccard_score(labels_phase, preds_phase, average=None)
    video_results_i = recognize_i.compute_video_AP('i')
    video_results_v = recognize_v.compute_video_AP('v')
    video_results_t = recognize_t.compute_video_AP('t')
    video_results_p = acc_p_meter.avg
    print(video_results_i["AP"])
    print(video_results_v["AP"])
    print(video_results_t["AP"])

    print("precision_each_phase:", precision_each_phase)
    print("recall_each_phase:", recall_each_phase)
    print("jaccard_each_phase:", jaccard_each_phase)
    print("precision_phase", precision_phase)
    print("recall_phase", recall_phase)
    print("jaccard_phase", jaccard_phase)
    print(f'[[test] [video_mAP_i: {video_results_i["mAP"]:.5f}, '
          f'video_mAP_v: {video_results_v["mAP"]:.5f}, video_mAP_t: {video_results_t["mAP"]:.5f},'
          f'video_acc_p:{video_results_p:.5f}]')


if __name__ == '__main__':
    ex.run_commandline()
