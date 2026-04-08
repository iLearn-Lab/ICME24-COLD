import torch
import torch.nn as nn
import torch.nn.functional as F


class CEncoder(nn.Module):
    def __init__(self, dim, z_dim, drop_rate):
        super(CEncoder, self).__init__()
        self.dim = dim
        self.latent_size = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.dim, 2 * self.latent_size),
            nn.ReLU(),
            # nn.LeakyReLU(0.01),
        )
        self.fc_mu = nn.Linear(2 * self.latent_size, self.latent_size)
        self.fc_std = nn.Linear(2 * self.latent_size, self.latent_size)

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)

    def forward(self, x):
        mu, std = self.encode(x)
        # x = self.dropout(x)

        return mu, std


class Encoder(nn.Module):
    def __init__(self, dim, z_dim, drop_rate):
        super(Encoder, self).__init__()

        self.cencoder = CEncoder(dim, z_dim, drop_rate=drop_rate)

    def forward(self, x):
        means, std = self.cencoder(x)

        eps = torch.randn(means.shape, device='cuda')
        z = means + eps * std

        return means, std, z


class MIEstimator(nn.Module):
    def __init__(self, dim):
        super(MIEstimator, self).__init__()
        self.linear = nn.Linear(dim, dim)  # project v onto q space

    def forward(self, v_means, v_std, zv, q_means, q_std, zq, h_labels, eps=1e-12):
        
        v_var = v_std.pow(2)  # square
        q_var = q_std.pow(2)
       
        kld_v = - 0.5 * (1 + v_var.log() - v_means.pow(2) - v_var).mean()
        kld_q = - 0.5 * (1 + q_var.log() - q_means.pow(2) - q_var).mean()

        kl1 = torch.log(q_std / (v_std + eps)) + (v_var + (v_means - q_means) ** 2) / (
                2 * v_var + eps) - 0.5
        kl2 = torch.log(v_std / (q_std + eps)) + (q_var + (q_means - v_means) ** 2) / (
                2 * q_var + eps) - 0.5

        # compute skl
        skl = 0.5 * (kl1 + kl2).mean()

        # Gradient for JSD mutual information estimation
        pos = (zv * h_labels.unsqueeze(dim=2)) + zq

        neg_mask = 1 - h_labels.unsqueeze(dim=2)
        neg = (zv * neg_mask) + zq

        loss1 = -F.softplus(-pos).mean() - F.softplus(neg).mean()

        loss = -loss1 + 0.1*skl +  0.1*kld_q +  0.1*kld_v
        # loss = -loss1 + 0.1 * skl
        # print('kld_v:' + str(kld_v.item()))
        # print('kld_q:' + str(kld_q.item()))
        # print('skl:' + str(skl.item()))
        # print('jsloss:' + str((-loss1).item()))
        # print('cross_query_video_loss:' + str(loss.item()))

        return loss
