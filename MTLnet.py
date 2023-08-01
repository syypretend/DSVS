import torch
import torch.nn as nn
from torchvision import models
from torchcrf import CRF


class MTLnet_lstm(nn.Module):
    def __init__(self, backbone, out_channel=1024, dropout=0.3, word_size=1024, use_crf=False):
        super(MTLnet_lstm, self).__init__()
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=True, progress=True)
            model_list = list(resnet.children())
            self.conv5 = nn.Sequential(*model_list[:-2])
            in_channel = self.conv5[-1][-2].bn3.num_features
        elif backbone == "resnet18":
            resnet = models.resnet18(pretrained=True, progress=True)
            model_list = list(resnet.children())
            self.conv5 = nn.Sequential(*model_list[:-2])
            in_channel = self.conv5[-1][-1].bn2.num_features
        self.in_channel = in_channel
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(num_tags=7, batch_first=True)
        self.out_channel = out_channel
        self.dropout = dropout
        self.i_conv1 = nn.Conv2d(in_channel, out_channel, 3)
        self.i_conv2 = nn.Conv2d(out_channel, 1024, 1)
        self.v_conv1 = nn.Conv2d(in_channel, out_channel, 3)
        self.v_conv2 = nn.Conv2d(out_channel * 2, 1024, 1)
        self.t_conv1 = nn.Conv2d(in_channel, out_channel, 3)
        self.t_conv2 = nn.Conv2d(out_channel * 2, 1024, 1)
        self.p_conv1 = nn.Conv2d(in_channel, out_channel, 3)
        self.p_conv2 = nn.Conv2d(out_channel, 1024, 1)
        self.pool5 = nn.AdaptiveAvgPool2d(1)
        self.label_embedding = nn.Embedding(num_embeddings=31, embedding_dim=word_size)
        self.transformer_decoder_ivt = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.out_channel, nhead=8, dim_feedforward=self.out_channel,
                                       dropout=0.3), num_layers=1)
        self.i_fc1 = nn.Linear(out_channel, 128)
        self.i_fc2 = nn.Linear(128, 6)
        self.v_fc1 = nn.Linear(out_channel, 128)
        self.v_fc2 = nn.Linear(128, 10)
        self.t_fc1 = nn.Linear(out_channel, 128)
        self.t_fc2 = nn.Linear(128, 15)

        # transformer_phase
        self.lstm = nn.LSTM(self.in_channel, 512, num_layers=2, batch_first=True, bidirectional=True)

        self.p_fc1 = nn.Linear(1024, 128)
        self.p_fc2 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, sequence_length, labels=None):
        batch_size, c, h, w = x.size()
        res5c = self.conv5(x)
        # instrument
        instrument = self.relu(self.i_conv1(res5c))
        instrument = self.relu(self.i_conv2(instrument))

        # verb
        verb = self.relu(self.v_conv1(res5c))
        verb = self.relu(self.v_conv2(torch.cat((verb, instrument), 1)))

        # target
        target = self.relu(self.t_conv1(res5c))
        target = self.relu(self.t_conv2(torch.cat((target, instrument), 1)))

        feature_i = self.pool5(instrument).view(instrument.size(0), -1).unsqueeze(1)
        feature_v = self.pool5(verb).view(verb.size(0), -1).unsqueeze(1)
        feature_t = self.pool5(target).view(target.size(0), -1).unsqueeze(1)
        feature = torch.cat((feature_i, feature_v, feature_t), 1)

        feature_i = feature[:, 0, :]
        feature_v = feature[:, 1, :]
        feature_t = feature[:, 2, :]

        # y_ivt: predict from fc
        feature_i = self.i_fc1(feature_i)
        pred_i = self.i_fc2(self.dropout(self.relu(feature_i)))
        feature_v = self.v_fc1(feature_v)
        pred_v = self.v_fc2(self.dropout(self.relu(feature_v)))
        feature_t = self.t_fc1(feature_t)
        pred_t = self.t_fc2(self.dropout(self.relu(feature_t)))

        # phase
        res = self.pool5(res5c).view(int(batch_size / sequence_length), sequence_length, -1)
        self.lstm.flatten_parameters()
        phase, _ = self.lstm(res)
        phase = phase.contiguous().view(-1, phase.shape[2])
        phase = self.dropout(self.relu(self.p_fc1(phase)))
        pred_p = self.p_fc2(phase)
        loss = None
        loss_fct = nn.CrossEntropyLoss().cuda()
        if labels is not None:
            if self.use_crf:
                loss = self.crf(pred_p.view(-1, sequence_length, 7), labels.view(-1, sequence_length).long())
                loss = -1 * loss
            else:
                loss = loss_fct(pred_p, labels.view(-1).long())
        else:
            print("Need labels to compute loss!")
        return pred_i, pred_v, pred_t, pred_p, res.view(-1, res.shape[-1]), loss

    def get_optim_policies(self, backbone_lr, subnet_lr, crf_lr):
        subnet_weight = list(self._flat([list(layer.parameters()) for layer in list(self.children())[2:]]))
        backbone_weight = list(self._flat(self.conv5.parameters()))
        crf_weight = list(self.crf.parameters())
        param = [
                 {'params': subnet_weight, 'lr': subnet_lr, 'name': 'subnet'},
                 {'params': backbone_weight, 'lr': backbone_lr, 'name': 'backbone'},
                 {'params': crf_weight, 'lr': crf_lr, 'name': 'crf'},
        ]
        return param

    def _flat(self, l):
        for k in l:
            if not isinstance(k, (list, tuple)):
                yield k
            else:
                yield from self._flat(k)