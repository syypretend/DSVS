import math
import os
import cv2
import imageio
from ivtmetrics.recognition import Recognition
import numpy as np
import torch
from tool.sequence_preprocess import get_train_idx, get_valid_idx
from tool.triplet2IVT import triplet2IVT
from tool.utils import AverageMeter, log_metrics, average_by_videos, cal_adj_matrix, weight_ivt
import pandas as pd
from tool.utils import sigmoid
import cvxpy as cp


def train(train_loader, model, loss, optimizer, epoch, map_path, config, ex, writer):
    loss_i_meter = AverageMeter()
    loss_v_meter = AverageMeter()
    loss_t_meter = AverageMeter()
    loss_p_meter = AverageMeter()
    loss_cross_meter = AverageMeter()
    loss_transition_meter = AverageMeter()
    loss_meter = AverageMeter()
    acc_p_meter = AverageMeter()
    recognize_i = Recognition(num_class=6)
    recognize_v = Recognition(num_class=10)
    recognize_t = Recognition(num_class=15)
    model.train()
    recognize_i.reset()
    recognize_v.reset()
    recognize_t.reset()
    for batch_idx, (frame_idx, v_name, input, target, target_p) in enumerate(train_loader):
        input = input.cuda()
        target_p = target_p.cuda()
        ivt_label = triplet2IVT(map_path, target)
        i_label, v_label, t_label = ivt_label[0].cuda(), ivt_label[1].cuda(), ivt_label[2].cuda()

        # ---- Train model ---- #
        pred_i, pred_v, pred_t, pred_p, _, transition_loss = model(input, config.sequence_length, target_p)
        pred_i = pred_i.view(-1, 6)
        pred_v = pred_v.view(-1, 10)
        pred_t = pred_t.view(-1, 15)
        
        # ---- Train loss ---- #
        i_loss = loss["i"](pred_i, i_label)
        v_loss = loss["v"](pred_v, v_label)
        t_loss = loss["t"](pred_t, t_label)
        cross_loss = loss["p"](pred_p, target_p.long())
        transition_loss = transition_loss.mean()

        initial_task_loss = 0
        if epoch <= 5:
            p_loss = transition_loss * 0.1
            weight = 0.3 * np.cos(1 / 60 * 2 * np.pi * epoch) + 0.5
            total_loss = weight * i_loss + (1 - weight) * p_loss
        else:
            p_loss = transition_loss * 0.1
            weight = 0.3 * np.cos(1 / 60 * 2 * np.pi * epoch) + 0.5
            total_loss = weight * (i_loss + v_loss + t_loss) + (1 - weight) * p_loss
        optimizer.zero_grad()
        total_loss.backward()
        # ---- optimizer ---- #
        optimizer.step()
        loss_meter.update(total_loss.item())
        loss_i_meter.update(i_loss.item())
        loss_v_meter.update(v_loss.item())
        loss_t_meter.update(t_loss.item())
        loss_p_meter.update(p_loss.item())
        loss_cross_meter.update(cross_loss.item())
        loss_transition_meter.update(transition_loss.item())

        # ---- Train mAP ---- #
        # mAP_i, mAP_v, mAP_t
        recognize_i.update(i_label.detach().cpu().numpy(), pred_i.detach().cpu().numpy())
        recognize_v.update(v_label.detach().cpu().numpy(), pred_v.detach().cpu().numpy())
        recognize_t.update(t_label.detach().cpu().numpy(), pred_t.detach().cpu().numpy())

        # acc_p
        target_p = target_p.detach().cpu()
        pred_labels = model.module.crf.decode(pred_p.view(-1, config.sequence_length, 7))
        pred_phase = torch.tensor(pred_labels).detach().cpu().view(-1)

        batch_corrects_phase = torch.sum(pred_phase == target_p)
        acc_p_meter.update(batch_corrects_phase)

        if (batch_idx + 1) % config.print_freq == 0:
            if config.grad_norm:
                print(f'[epoch {epoch}][batch {batch_idx + 1}/{len(train_loader)}]'
                      f'[train_loss:{loss_meter.avg:.5f},loss_i:{loss_i_meter.avg:.5f},'
                      f'loss_v:{loss_v_meter.avg:.5f},loss_t:{loss_t_meter.avg:.5f},loss_p:{loss_p_meter.avg:.5f},'
                      f'loss_cross:{loss_cross_meter.avg:.5f},loss_transition:{loss_transition_meter.avg:.3f}]'
                      f'[weight:{model.module.weights.detach().cpu().numpy()}]')
            else:
                print(f'[epoch {epoch}][batch {batch_idx + 1}/{len(train_loader)}]'
                      f'[train_loss:{loss_meter.avg:.5f},loss_i:{loss_i_meter.avg:.5f},'
                      f'loss_v:{loss_v_meter.avg:.5f},loss_t:{loss_t_meter.avg:.5f},loss_p:{loss_p_meter.avg:.5f},'
                      f'loss_cross:{loss_cross_meter.avg:.5f},loss_transition:{loss_transition_meter.avg:.3f}]')
    result_i = recognize_i.compute_AP("i")
    result_v = recognize_v.compute_AP("v")
    result_t = recognize_t.compute_AP('t')
    print("===========================================================================================")
    print(
        f'[epoch {epoch}] [train] [mAP_i: {result_i["mAP"]:.5f}, mAP_v: {result_v["mAP"]:.5f}, mAP_t: {result_t["mAP"]:.5f}, '
        f'acc_p: {acc_p_meter.sum / (len(train_loader) * config.batch_size):.5f}]')
    metrics = {
        'train_loss': {
            "total_train_loss": loss_meter.avg,
            'loss_i': loss_i_meter.avg,
            'loss_v': loss_v_meter.avg,
            'loss_t': loss_t_meter.avg,
            "loss_p": loss_p_meter.avg,
            "loss_cross": loss_cross_meter.avg,
            "loss_transition": loss_transition_meter.avg
        },
        "train_metrics": {
            "mAP_i": result_i["mAP"],
            "mAP_v": result_v["mAP"],
            "mAP_t": result_t["mAP"],
            "acc_p": acc_p_meter.sum / (len(train_loader) * config.batch_size)
        }
    }
    log_metrics(metrics, epoch, ex, writer)
    return initial_task_loss


def valid(valid_loader, model, loss, epoch, map_path, config, ex, writer):
    loss_i_meter = AverageMeter()
    loss_v_meter = AverageMeter()
    loss_t_meter = AverageMeter()
    loss_p_meter = AverageMeter()
    loss_cross_meter = AverageMeter()
    loss_transition_meter = AverageMeter()
    loss_meter = AverageMeter()
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
    model.eval()
    with torch.no_grad():
        for batch_idx, (frame_idx, v_name, input, target, target_p) in enumerate(valid_loader):
            input = input.cuda()
            target_p = target_p.cuda()
            ivt_label = triplet2IVT(map_path, target)
            i_label, v_label, t_label = ivt_label[0].cuda(), ivt_label[1].cuda(), ivt_label[2].cuda()

            # ---- valid model ---- #
            pred_i, pred_v, pred_t, pred_p, _, transition_loss = model(input, config.sequence_length, target_p)
            
            pred_i = pred_i.view(-1, 6)
            pred_v = pred_v.view(-1, 10)
            pred_t = pred_t.view(-1, 15)
            pred_p = pred_p.view(-1, 7)[config.sequence_length - 1::config.sequence_length]
            target_p = target_p[config.sequence_length - 1::config.sequence_length].cuda()

            # ---- valid loss ---- #
            i_loss = loss["i"](pred_i, i_label)
            v_loss = loss["v"](pred_v, v_label)
            t_loss = loss["t"](pred_t, t_label)
            cross_loss = loss["p"](pred_p, target_p.long())
            transition_loss = transition_loss.mean()
            if epoch <= 5:
                p_loss = transition_loss * 0.1
                weight = 0.3 * np.cos(1 / 60 * 2 * np.pi * epoch) + 0.5
                total_loss = weight * i_loss + (1 - weight) * p_loss
            else:
                p_loss = transition_loss * 0.1
                weight = 0.3 * np.cos(1 / 60 * 2 * np.pi * epoch) + 0.5
                total_loss = weight * (i_loss + v_loss + t_loss) + (1 - weight) * p_loss

            loss_meter.update(total_loss.item())
            loss_i_meter.update(i_loss.item())
            loss_v_meter.update(v_loss.item())
            loss_t_meter.update(t_loss.item())
            loss_p_meter.update(p_loss.item())
            loss_cross_meter.update(cross_loss.item())
            loss_transition_meter.update(transition_loss.item())

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

    print(f'[epoch {epoch}] '
          f'[valid_loss:{loss_meter.avg:.5f},loss_i:{loss_i_meter.avg:.5f},'
          f'loss_v:{loss_v_meter.avg:.5f},loss_t:{loss_t_meter.avg:.5f},loss_p:{loss_p_meter.avg:.5f},'
          f'loss_cross:{loss_cross_meter.avg:.5f},loss_transition:{loss_transition_meter.avg:.5f}]')
    
    # compute mAP by each video without sequence_length
    # each video
    i_preds = np.concatenate(i_pre, axis=0)
    v_preds = np.concatenate(v_pre, axis=0)
    t_preds = np.concatenate(t_pre, axis=0)
    p_preds = np.concatenate(p_pre, axis=0)
    i_target = np.concatenate(i_tar, axis=0)
    v_target = np.concatenate(v_tar, axis=0)
    t_target = np.concatenate(t_tar, axis=0)
    p_target = np.concatenate(p_tar, axis=0)
    each_video_length = get_valid_idx(config)[2]
    count = 0
    recognize_i.reset_global()
    recognize_v.reset_global()
    recognize_t.reset_global()
    # each video
    for i in range(len(each_video_length)):
        video_i_prediction = []
        video_v_prediction = []
        video_t_prediction = []
        video_p_prediction = []
        video_i_target = []
        video_v_target = []
        video_t_target = []
        video_p_target = []
        if not config.DEBUG:
            video_name = [config.video_names[j] for j in config.valid_index][i]
        else:
            video_name = [config.video_names[j] for j in config.valid_index[:2]][i]
        if i == len(each_video_length) - 1:
            video_length = int((len(valid_loader) * config.batch_size - count) / config.sequence_length)

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
        recognize_i.update(i_target_np, i_preds_np)
        recognize_i.video_end()
        recognize_v.update(v_target_np, v_preds_np)
        recognize_v.video_end()
        recognize_t.update(t_target_np, t_preds_np)
        recognize_t.video_end()
        

        pred_labels = model.module.crf.decode(torch.from_numpy(p_preds_np).view(1, -1, 7).cuda())
        pred_phase = torch.tensor(pred_labels).detach().cpu().view(-1)
        video_corrects_phase = torch.sum(pred_phase == torch.from_numpy(p_target_np).view(-1))
        acc_p_meter.update(video_corrects_phase / len(p_target_np))
    video_results_i = recognize_i.compute_video_AP('i')
    video_results_v = recognize_v.compute_video_AP('v')
    video_results_t = recognize_t.compute_video_AP('t')
    video_results_p = acc_p_meter.avg
    print(f'[epoch {epoch}] [valid] [mAP_i: {video_results_i["mAP"]:.5f}, '
          f'mAP_v: {video_results_v["mAP"]:.5f}, mAP_t: {video_results_t["mAP"]:.5f}, '
          f'acc_p:{video_results_p:.5f}]')
    print("===========================================================================================")
    metrics = {
        'valid_loss': {
            "total_valid_loss": loss_meter.avg,
            'loss_i': loss_i_meter.avg,
            'loss_v': loss_v_meter.avg,
            'loss_t': loss_t_meter.avg,
            "loss_p": loss_p_meter.avg,
            "loss_cross": loss_cross_meter.avg,
            "loss_transition": loss_transition_meter.avg
        },
        'valid_metric': {
            "mAP_i": video_results_i["mAP"],
            "mAP_v": video_results_v["mAP"],
            "mAP_t": video_results_t["mAP"],
            "acc_p": acc_p_meter.avg
        }
    }
    log_metrics(metrics, epoch, ex, writer)
    return metrics["valid_metric"]
