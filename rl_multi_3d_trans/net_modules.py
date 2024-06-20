# from modules import SAB, PMA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


## trans for everything, then query with self
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)


    def forward(self, Q, K, mask=None):
        q = Q.clone()
        k_o = K.clone()
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Computing the raw attention scores
        attention_scores = Q_.bmm(K_.transpose(1, 2)) / np.sqrt(self.dim_V)

        # Applying the mask - setting masked positions to a large negative value to zero their weights in the softmax
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).repeat(1, self.num_heads, 1).view(mask.size()[0] * self.num_heads, 1,
                                                                                mask.size()[-1])
            attention_scores = attention_scores.masked_fill(mask_expanded == True, float('-inf'))
            # attention_scores = attention_scores.masked_fill(mask == True, float('-inf'))

        # Normalizing the attention scores to probabilities
        A = torch.softmax(attention_scores, dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O



class FcModule(nn.Module):
    def __init__(self, net_width=256):
        super().__init__()
        self.net_width = net_width
        self.fc0_2 = nn.Linear(int(net_width * 2), net_width)
        self.fc0_4 = nn.Linear(int(net_width * 4), net_width)
        self.fc0_8 = nn.Linear(int(net_width * 8), net_width)
        self.fc1_1 = nn.Linear(net_width, int(net_width / 2))
        self.bn1 = nn.BatchNorm1d(int(net_width / 2))
        self.fc1_2 = nn.Linear(int(net_width / 2), net_width)
        self.fc2_1 = nn.Linear(net_width, int(net_width / 2))
        self.bn2 = nn.BatchNorm1d(int(net_width / 2))
        self.fc2_2 = nn.Linear(int(net_width / 2), net_width)

    def forward(self, x, times=2):

        if x.size(-1) <= self.net_width * 2:
            padSize = self.net_width * 2 - x.size(-1)
            x = self.fc0_2(F.pad(x, (0, padSize)))
        elif x.size(-1) <= self.net_width * 4:
            padSize = self.net_width * 4 - x.size(-1)
            x = self.fc0_4(F.pad(x, (0, padSize)))
        elif x.size(-1) <= self.net_width * 8:
            padSize = self.net_width * 8 - x.size(-1)
            x = self.fc0_8(F.pad(x, (0, padSize)))
        input1 = x
        x = self.fc1_1(x)
        x = self.bn1(x)
        x = F.relu(self.fc1_2(x)) + input1
        ##############  add-on 11/27
        # if times > 1:
        #     input2 = x
        #     x = self.fc2_1(x)
        #     x = self.bn2(x)
        #     x = F.relu(self.fc2_2(x)) + input2
        ###################3

        input2 = x
        x = self.fc2_1(x)
        x = self.bn2(x)
        x = F.relu(self.fc2_2(x)) + input2
        return x


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


class LayerNormEmbedding(nn.Module):
    def __init__(self, input_dim_pad=32, hidden=64, output_dim=128, higher=False):
        super(LayerNormEmbedding, self).__init__()
        self.input_dim_pad = input_dim_pad
        self.fc32 = nn.Linear(32, hidden)
        self.fc64 = nn.Linear(64, hidden)
        self.fc128 = nn.Linear(128, hidden)
        self.bn1 = nn.LayerNorm(hidden)
        self.fc11 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, output_dim)
        self.act = nn.ReLU()
        self.res_fc = nn.Linear(hidden, output_dim) if hidden != output_dim else None
        self.output_dim = output_dim

    def forward(self, input_tensor, position_index=-1, max_len=3):
        # Padding or truncating the input tensor
        # assert input_tensor.size(-1) <= self.input_dim_pad * 2

        for i in range(3):
            multplier = 2 ** i
            if input_tensor.size(-1) <= self.input_dim_pad * multplier:
                padSize = self.input_dim_pad * multplier
                break
        padded_input = F.pad(input_tensor, (0, padSize - input_tensor.size(-1)))

        input_dims = len(input_tensor.shape)
        if input_dims == 3:
            padded_input = padded_input.view(-1, padSize)
        if multplier == 1:
            x = self.bn1(self.fc32(padded_input))
        elif multplier == 2:
            x = self.fc64(padded_input)
            x = self.bn1(x)
        elif multplier == 4:
            x = self.bn1(self.fc128(padded_input))

        x = F.relu(self.fc11(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        if position_index >= 0:
            pos_encoding = positional_encoding(max_len, self.output_dim)
            pos_encoding = pos_encoding[:, position_index, :].to(input_tensor.device)
            x = x + pos_encoding
        if input_dims == 3:
            x = x.view(input_tensor.shape[0], -1, self.output_dim)
        elif input_dims == 2:
            x = x.unsqueeze(1)
        return x


class BatchNormEmbedding(nn.Module):
    def __init__(self, input_dim_pad=32, hidden=64, output_dim=128, higher=False):
        super(BatchNormEmbedding, self).__init__()
        self.input_dim_pad = input_dim_pad
        self.fc32 = nn.Linear(32, hidden)
        self.fc64 = nn.Linear(64, hidden)
        self.fc128 = nn.Linear(128, hidden)
        if higher:
            self.fc256 = nn.Linear(256, hidden)
            self.fc512 = nn.Linear(512, hidden)
            self.fc1024 = nn.Linear(1024, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc11 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, output_dim)
        self.act = nn.ReLU()
        self.res_fc = nn.Linear(hidden, output_dim) if hidden != output_dim else None
        self.output_dim = output_dim

    def forward(self, input_tensor, position_index=-1, max_len=3):
        # Padding or truncating the input tensor
        # assert input_tensor.size(-1) <= self.input_dim_pad * 2

        for i in range(4):
            multplier = 2 ** i
            if input_tensor.size(-1) <= self.input_dim_pad * multplier:
                padSize = self.input_dim_pad * multplier
                break
        padded_input = F.pad(input_tensor, (0, padSize - input_tensor.size(-1)))

        input_dims = len(input_tensor.shape)
        if input_dims == 3:
            padded_input = padded_input.view(-1, padSize)
        if multplier == 1:
            x = self.bn1(self.fc32(padded_input))
        elif multplier == 2:
            x = self.fc64(padded_input)
            x = self.bn1(x)
        elif multplier == 4:
            x = self.bn1(self.fc128(padded_input))
        elif multplier == 8:
            x = self.bn1(self.fc256(padded_input))
        elif multplier == 16:
            x = self.bn1(self.fc512(padded_input))
        elif multplier == 32:
            x = self.bn1(self.fc1024(padded_input))

        x = F.relu(self.fc11(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        if position_index >= 0:
            pos_encoding = positional_encoding(max_len, self.output_dim)
            pos_encoding = pos_encoding[:, position_index, :].to(input_tensor.device)
            x = x + pos_encoding
        if input_dims == 3:
            x = x.view(input_tensor.shape[0], -1, self.output_dim)
        elif input_dims == 2:
            x = x.unsqueeze(1)
        return x


class Embedding(nn.Module):
    def __init__(self, input_dim_pad=32, hidden=64, output_dim=128):
        super(Embedding, self).__init__()
        self.input_dim_pad = input_dim_pad
        self.fc32 = nn.Linear(32, hidden)
        self.fc64 = nn.Linear(64, hidden)
        self.fc128 = nn.Linear(128, hidden)
        self.fc256 = nn.Linear(256, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc11 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.act = nn.ReLU()
        self.res_fc = nn.Linear(hidden, output_dim) if hidden != output_dim else None
        self.output_dim = output_dim

    def forward(self, input_tensor, position_index=-1, max_len=3):
        # Padding or truncating the input tensor
        # assert input_tensor.size(-1) <= self.input_dim_pad * 2

        for i in range(4):
            multplier = 2 ** i
            if input_tensor.size(-1) <= self.input_dim_pad * multplier:
                padSize = self.input_dim_pad * multplier
                break
        padded_input = F.pad(input_tensor, (0, padSize - input_tensor.size(-1)))

        input_dims = len(input_tensor.shape)
        if input_dims == 3:
            padded_input = padded_input.view(-1, padSize)
        if multplier == 1:
            x = self.bn1(self.fc32(padded_input))
        elif multplier == 2:
            x = self.bn1(self.fc64(padded_input))
        elif multplier == 4:
            x = self.bn1(self.fc128(padded_input))
        elif multplier == 8:
            x = self.bn1(self.fc256(padded_input))

        x = F.relu(self.fc11(x))
        x = self.bn2(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        if position_index >= 0:
            pos_encoding = positional_encoding(max_len, self.output_dim)
            pos_encoding = pos_encoding[:, position_index, :].to(input_tensor.device)
            x = x + pos_encoding
        if input_dims == 3:
            x = x.view(input_tensor.shape[0], -1, self.output_dim)
        elif input_dims == 2:
            x = x.unsqueeze(1)
        return x


class Embedding_Res(nn.Module):
    def __init__(self, input_dim_pad=32, hidden=64, output_dim=128):
        super(Embedding_Res, self).__init__()
        self.input_dim_pad = input_dim_pad
        self.fc32 = nn.Linear(32, 128)
        self.fc64 = nn.Linear(64, 128)
        self.fc128 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc11 = nn.Linear(128, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.fc3 = nn.Linear(hidden, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_dim)
        self.act = nn.ReLU()
        self.res_fc = nn.Linear(hidden, output_dim) if hidden != output_dim else None
        self.output_dim = output_dim

    def forward(self, input_tensor, position_index=-1, max_len=3):
        # Padding or truncating the input tensor
        # assert input_tensor.size(-1) <= self.input_dim_pad * 2

        for i in range(3):
            multplier = 2 ** i
            if input_tensor.size(-1) <= self.input_dim_pad * multplier:
                padSize = self.input_dim_pad * multplier
        padded_input = F.pad(input_tensor, (0, padSize - input_tensor.size(-1)))

        input_dims = len(input_tensor.shape)
        if input_dims == 3:
            padded_input = padded_input.view(-1, padSize)
        if multplier == 1:
            x = self.bn1(self.fc32(padded_input))
        elif multplier == 2:
            x = self.bn1(self.fc64(padded_input))
        elif multplier == 4:
            x = self.bn1(self.fc128(padded_input))

        identity = x
        x = self.fc11(x)
        x = self.fc2(x)
        x = self.bn3(self.fc3(x))
        x = F.relu(x + identity)
        x = self.fc4(x)

        # x = self.fc4(x)

        if position_index >= 0:
            pos_encoding = positional_encoding(max_len, self.output_dim)
            pos_encoding = pos_encoding[:, position_index, :].to(input_tensor.device)
            x = x + pos_encoding
        if input_dims == 3:
            x = x.view(input_tensor.shape[0], -1, self.output_dim)
        elif input_dims == 2:
            x = x.unsqueeze(1)
        return x
