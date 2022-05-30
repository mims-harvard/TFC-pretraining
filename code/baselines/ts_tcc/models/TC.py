import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer
import torch.nn.functional as F


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)


class convEncoder(nn.Module):
    def __init__(self, configs, device, kernel_sizes, num_channels):
        super(convEncoder, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_heads = len(kernel_sizes)
        self.num_channels = num_channels
        self.feature_len = num_channels // self.num_heads
        self.device = device
        self.conv_Q_encoders = nn.ModuleList([nn.Conv1d(num_channels, self.feature_len, kernel_size=n, padding='same') for n in self.kernel_sizes])
        self.conv_V_encoders = nn.ModuleList([nn.Conv1d(num_channels, self.feature_len, kernel_size=n, padding='same') for n in self.kernel_sizes])
        self.conv_K_encoders = nn.ModuleList([nn.Conv1d(num_channels, self.feature_len, kernel_size=n, padding='same') for n in self.kernel_sizes])
        self.dim = np.sqrt(configs.window_len)

    def forward(self, X):
        heads_out = []
        for Qe, Ve, Ke in zip(self.conv_Q_encoders, self.conv_V_encoders, self.conv_K_encoders):
            Q = Qe(X)
            V = Ve(X)
            K = Ke(X)
            score = torch.bmm(Q.transpose(1,2), K) / self.dim #             K, Q, V of shape batch_size (nb) * feature_len (fl) * window size/time steps (ts)
            attn = F.softmax(score, -1)                       #             Q.T = nb * ts * fl ; K = nb * fl * ts, score = nb * ts * ts
            context = torch.bmm(attn, V.transpose(1,2)).transpose(1,2) # nb * fl * ts, same as QVK
            heads_out.append(context) # list of num_heads tensors of shape nb * fl * ts
        return torch.cat(heads_out, dim=1)

class TS_SD(nn.Module):
    def __init__(self, configs, device, kernel_sizes, num_channels, num_encoders):
        super(TS_SD, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_heads = len(kernel_sizes)
        self.num_channels = num_channels
        self.feature_len = num_channels // self.num_heads
        self.device = device  # conv : nb * n_ic * ws -> nb * n_feat * ws
        self.conv_encoders = nn.ModuleList([convEncoder(configs, device, kernel_sizes, num_channels) for _ in range(num_encoders)])
        self.preencoder = nn.Linear(configs.input_channels, num_channels)
        self.linear = nn.Linear(num_channels, 1)
        self.final_conv_1 = nn.Conv1d(num_channels, 32, kernel_size=8, stride=4)
        self.final_conv_2 = nn.Conv1d(32, 64, kernel_size=8, stride=4)
        self.final_conv_3 = nn.Conv1d(64, self.feature_len, kernel_size=8, stride=4)
        self.logit = nn.Linear(8, configs.num_classes) #176, 8, or 624

    def forward(self, signal, mode="pretrain"):
        X = self.preencoder(signal.transpose(1,2)).transpose(1,2) # nb * 64 * 1500
        for encoder in self.conv_encoders:
            X = encoder(X)

        if mode=='pretrain': # nb * (fl * num_heads) * ts
            return self.linear(X.transpose(1,2)).transpose(1,2)
        else:
            final_conv = self.final_conv_3(self.final_conv_2(self.final_conv_1(X)))
            flat = torch.reshape(final_conv, (final_conv.shape[0], -1))
            return self.logit(flat)

