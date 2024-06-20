# from modules import SAB, PMA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from air_corridor.tools.util import nan_recoding
from rl_multi_3d_trans.net_modules import FcModule, Embedding, MAB, BatchNormEmbedding, LayerNormEmbedding


## whcih is actually 10_3

class SmallSetTransformer(nn.Module):
    def __init__(self, neighbor_dimension=7, net_width=256, with_position=False, token_query=False, num_enc=4,
                 num_dec=2, logger=None):
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, 1, net_width))
        nn.init.xavier_uniform_(self.S)
        encoder_layer = nn.TransformerEncoderLayer(d_model=net_width, nhead=4, dim_feedforward=net_width * 2,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc)
        self.decoder_mab = MAB(net_width, net_width, net_width, num_heads=4, ln=True)

        self.decoder = nn.ModuleList(
            [MAB(net_width, net_width, net_width, num_heads=4, ln=True) for _ in range(num_dec)])
        self.eb = BatchNormEmbedding(output_dim=net_width, hidden=128)
        self.fc = nn.Linear(net_width, net_width)
        self.with_position = with_position
        self.token_query = token_query
        self.fc1 = nn.Linear(2 * net_width, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.fc3 = nn.Linear(net_width, net_width)
        self.logger = logger
        self.fc_module = FcModule(net_width=net_width)

        ########################################
        # self.bn1 = nn.BatchNorm1d(net_width)
        # self.bn2 = nn.BatchNorm1d(net_width)
        # self.fc3 = nn.Linear(net_width, net_width)
        # ########################################

    def forward(self, x, uav, corridor):
        x1 = self.encoder(x)
        nan_recoding(self.logger, x1, 'encoding')

        c = corridor.view(corridor.size(0), -1)
        state = torch.cat([uav, c], axis=1)

        query = self.eb(state)
        for layer in self.decoder:
            query = layer.forward(query, x1)
        x7 = query



        x7 = x7.view(x7.size(0), -1)
        x8 = self.fc_module(x7)
        return x8


class FixedBranch(nn.Module):
    def __init__(self, input_dimension=11, net_width=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class BetaActorMulti(nn.Module):
    def __init__(self, s1_dim, s2_dim, action_dim, net_width, shared_layers=None, beta_base=1.0):
        super(BetaActorMulti, self).__init__()
        self.fc1 = nn.Linear(net_width, net_width)
        self.fc2_a = nn.Linear(net_width, int(net_width / 2))
        self.bn1 = nn.BatchNorm1d(int(net_width / 2))
        self.fc2_b = nn.Linear(int(net_width / 2), net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)
        if shared_layers is None:
            self.intput_merge = MergedModel(s1_dim, s2_dim, net_width)
        else:
            self.intput_merge = shared_layers
        self.beta_base = beta_base + 1e-3

    def forward(self, s1, s2):
        # merged_input = self.intput_merge(s1, s2)
        # alpha = F.softplus(self.alpha_head(merged_input)) + 1.0
        # beta = F.softplus(self.beta_head(merged_input)) + 1.0

        merged_input = self.intput_merge(s1, s2)
        x = F.relu(self.fc1(merged_input))
        x_a = F.relu(self.fc2_a(x))
        x_b = F.relu(self.fc2_b(x_a)) + x
        alpha = F.softplus(self.alpha_head(x_b)) + self.beta_base
        beta = F.softplus(self.beta_head(x_b)) + self.beta_base
        return alpha, beta

    def get_dist(self, s1, s2):
        alpha, beta = self.forward(s1, s2)
        dist = Beta(alpha, beta)
        return dist, alpha, beta

    def dist_mode(self, s1, s2):
        alpha, beta = self.forward(s1, s2)
        #  mode is the value that appears most often
        mode = (alpha - 1) / (alpha + beta - 2)
        return mode


class CriticMulti(nn.Module):
    def __init__(self, s1_dim, s2_dim, net_width, shared_layers=None):
        super(CriticMulti, self).__init__()
        self.C4 = nn.Linear(net_width, 1)
        if shared_layers is None:
            self.intput_merge = MergedModel(s1_dim, s2_dim, net_width)
        else:
            self.intput_merge = shared_layers

    def forward(self, s1, s2):
        merged_input = self.intput_merge(s1, s2)
        v = self.C4(merged_input)
        return v


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    return pe


class MergedModel(nn.Module):
    def __init__(self, s1_dim, s2_dim, net_width, with_position, token_query, num_enc, num_dec, logger=None):
        super(MergedModel, self).__init__()
        # self.fixed_branch = FixedBranch(s1_dim, net_width)
        self.trans = SmallSetTransformer(net_width, net_width, with_position, token_query, num_enc, num_dec, logger)
        self.net_width = net_width
        self.eb1 = BatchNormEmbedding(output_dim=net_width)
        self.eb2 = BatchNormEmbedding(output_dim=net_width)
        self.eb3 = BatchNormEmbedding(output_dim=net_width)


        self.with_position = with_position
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, s1, s2=None):
        s1_p = self.eb1(s1)
        s3 = s2[:, -4:]
        s2 = s2[:, :-4]

        s2_p = self.eb2(s2)
        s3_p = self.eb3(s3)
        # s2_p = self.eb_with_skip(s2, self.eb2)
        # s3_p = self.eb_with_skip(s3, self.eb3)

        s_p = torch.cat([s1_p, s2_p, s3_p], axis=1)
        c = s2[:, -3:]

        x = self.trans(s_p, uav=s1, corridor=c)
        nan_recoding(self.logger, x, 'trans_output')
        return x
