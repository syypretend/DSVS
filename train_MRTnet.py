# import setproctitle
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
from MRTnet_trainer import train, valid
from dataset import CholecVideo
from experiment.ex_train_MRTnet import ingredient, parse_config
from model.MRTnet import MRTnet_lstm
# sacred
from tool.sequence_preprocess import SeqSampler, get_train_idx, get_valid_idx
from tool.utils import calc_weight_loss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ex = Experiment('train_MRTnet', save_git_info=False, ingredients=[ingredient])
ex.observers.append(FileStorageObserver('log/sacred/train_MRTnet'))

# tensorboard
from torch.utils.tensorboard import SummaryWriter
import time

local_time = time.localtime()
time_str = f'{local_time.tm_year}-{local_time.tm_mon:02}-{local_time.tm_mday:02}--' \
           f'{local_time.tm_hour:02}-{local_time.tm_min:02}-{local_time.tm_sec:02}'
log_dir = f'log/tensorboard/train_MRTnet/{time_str}'
writer = SummaryWriter(log_dir)
print(f'use `$ tensorboard --logdir={log_dir} --bind_all` to open tensorboard')


@ex.main
def train_MRTnet(_config):
    config = parse_config(_config)
    maps_dict = np.genfromtxt(config.map_path, dtype=int, comments='#', delimiter=',', skip_header=0)
    train_transform = transforms.Compose([
        transforms.Resize((256, 448)),
        # transforms.RandomCrop(224),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.GaussianBlur(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((256, 448)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    torch.backends.cudnn.benchmark = True

    if not config.DEBUG:
        train_dataset = CholecVideo(
            data_dir=config.data_dir,
            triplet_dir=config.triplet_dir,
            phase_dir=config.phase_dir,
            sequence_length=config.sequence_length,
            video_names=[config.video_names[i] for i in config.train_index],
            transform=train_transform,
            batch_size=config.batch_size
        )
        train_video_name = [config.video_names[i] for i in config.train_index]
    else:
        # DEBUG (small dataset)
        train_dataset = CholecVideo(
            data_dir=config.data_dir,
            triplet_dir=config.triplet_dir,
            phase_dir=config.phase_dir,
            sequence_length=config.sequence_length,
            video_names=[config.video_names[i] for i in config.train_index[:2]],
            transform=train_transform,
            batch_size=config.batch_size
        )
        train_video_name = [config.video_names[i] for i in config.train_index[:2]]
    if not config.DEBUG:
        valid_dataset = CholecVideo(
            data_dir=config.data_dir,
            triplet_dir=config.triplet_dir,
            phase_dir=config.phase_dir,
            sequence_length=config.sequence_length,
            video_names=[config.video_names[i] for i in config.valid_index],
            transform=valid_transform,
            batch_size=config.batch_size
        )
        valid_video_name = [config.video_names[i] for i in config.valid_index]
    else:
        # DEBUG (small dataset)
        valid_dataset = CholecVideo(
            data_dir=config.data_dir,
            triplet_dir=config.triplet_dir,
            phase_dir=config.phase_dir,
            sequence_length=config.sequence_length,
            video_names=[config.video_names[i] for i in config.valid_index[:2]],
            transform=valid_transform,
            batch_size=config.batch_size
        )
        valid_video_name = [config.video_names[i] for i in config.valid_index[:2]]

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True,
                              num_workers=config.num_workers,
                              sampler=SeqSampler(train_dataset, get_train_idx(config)[1]),
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True,
                              num_workers=config.num_workers,
                              sampler=SeqSampler(valid_dataset, get_valid_idx(config)[1]),
                              drop_last=True)
    # ==================== Model ====================
    net = MRTnet_lstm(backbone="resnet18", use_crf=True)
    model = nn.DataParallel(net).cuda()
    # =================== Loss Function =============
    if config.loss_weight_file is None:
        print('Start calculating ivt loss weight.')
        weight = calc_weight_loss(train_loader, maps_dict)
        np.save('data/loss_weight.npy', weight.detach().cpu().numpy())
        ex.add_artifact('data/loss_weight.npy')
    else:
        print(f'Load ivt loss weight from {config.loss_weight_file}.')
        ex.add_resource(config.loss_weight_file)
        weight = np.load(config.loss_weight_file)
        weight = torch.from_numpy(weight)

    loss = {}
    loss['i'] = nn.BCEWithLogitsLoss(weight[:6]).cuda()  # i
    loss['v'] = nn.BCEWithLogitsLoss(weight[6: 6 + 10]).cuda()  # v
    loss['t'] = nn.BCEWithLogitsLoss(weight[6 + 10:]).cuda()  # t
    loss['p'] = nn.CrossEntropyLoss().cuda()
    # ===================== Optimizer ===============
    policies = net.get_optim_policies(backbone_lr=config.backbone_lr, subnet_lr=config.subnet_lr, crf_lr=config.crf_lr)
    if config.optim == 'sgd':
        optimizer = torch.optim.SGD(policies, weight_decay=1e-5, momentum=0.9)
    elif config.optim == 'adam':
        optimizer = torch.optim.AdamW(policies, weight_decay=1e-5, eps=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, config.scheduler_gamma)

    # ========================== resume ==========================
    if config.checkpoint is not None:
        start_epoch = resume(config.checkpoint, net, optimizer, scheduler)
        model = nn.DataParallel(net).cuda()
        ex.add_resource(config.checkpoint)
    else:
        start_epoch = 0
    # ==================== training ====================
    print("MRTnet starts training:")
    for epoch in range(start_epoch + 1, config.epochs + 1):
        initial_task_loss = train(train_loader, model, loss, optimizer, epoch, config.map_path, config, ex, writer)
        scheduler.step()
        metrics = valid(valid_loader, model, loss, epoch, config.map_path, config, ex, writer)
        save_checkpoint(f'data/MRTnet_checkpoint_{epoch}.pth', epoch, model, optimizer, initial_task_loss, scheduler,
                        metrics=metrics, ex=ex)


def save_checkpoint(path, epoch, model, optimizer, initial_task_loss, scheduler=None, metrics=None, ex=None):
    save_dict = {'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch, "initial_task_loss": initial_task_loss}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    if metrics is not None:
        for k, v in metrics.items():
            save_dict[k] = v
    torch.save(save_dict, path)
    if ex is not None:
        ex.add_artifact(path)


def resume(path, model, optimizer, scheluder=None, ex=None):
    point_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(point_dict['model_state_dict'])
    optimizer.load_state_dict(point_dict['optimizer_state_dict'])
    if scheluder is not None and 'scheduler_state_dict' in point_dict.keys():
        scheluder.load_state_dict(point_dict['scheduler_state_dict'])
    if ex is not None:
        ex.add_resource(path)
    print(f"Load checkpoint from {path}, epoch: {point_dict['epoch']}")
    return point_dict['epoch']


if __name__ == '__main__':
    ex.run_commandline()
