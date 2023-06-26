import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from parms_setting import settings
args = settings()


def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1 = None, s_bias2 = None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class CSGLMD(nn.Module):    #
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout):
        super(CSGLMD, self).__init__()

        self.encoder_o1 = GCNConv(feature, hidden1)
        self.prelu1_o1 = nn.PReLU(hidden1)
        self.encoder_o2 = GCNConv(hidden1, hidden2)
        self.prelu_o2 = nn.PReLU(hidden2)

        self.mlp = torch.nn.ModuleList()
        for i in range(1):
            self.mlp.append(nn.Linear(hidden2, hidden2))

        self.mlp1 = nn.Linear(hidden2, hidden2)
        self.decoder1 = nn.Linear(hidden2 * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)


        self.disc = Discriminator(hidden2)

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, data_o, data_a, idx):
        x_o, adj = data_o.x, data_o.edge_index

        x_a = data_a.x

        # original graph encoder
        x1_o = self.prelu1_o1(self.encoder_o1(x_o, adj))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)

        x2_o = self.encoder_o2(x1_o, adj)
        x2_o = self.prelu_o2(x2_o)

        # corrupt graph encoder
        x1_o_a = self.prelu1_o1(self.encoder_o1(x_a, adj))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)

        x2_o_a = self.encoder_o2(x1_o_a, adj)
        x2_o_a = self.prelu_o2(x2_o_a)  #

        # graph level representation
        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)
        h_os = self.mlp1(h_os)

        h_os_a = self.read(x2_o_a)
        h_os_a = self.sigm(h_os_a)
        h_os_a = self.mlp1(h_os_a)
        # print("h: ",h_os.shape, h_os_a.shape)

        # Adversarial learning
        sc_1 = x2_o.squeeze(0)
        sc_2 = x2_o_a.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)
        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)
        logits = torch.cat((sc_1, sc_2),1)  #1,2280

        # contrastive learning
        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os_a, x2_o_a, x2_o)
        if args.task_type == 'LDA':
            entity1 = x2_o[idx[0]]
            entity2 = x2_o[idx[1] + 240]
            # dataset2
            # entity1 = x2_o[idx[0]]
            # entity2 = x2_o[idx[1] + 665]

        if args.task_type == 'MDA':
            #dataset1
            entity1 = x2_o[idx[0] + 645]
            entity2 = x2_o[idx[1] + 240]

            # dataset2
            # entity1 = x2_o[idx[0] + 981]
            # entity2 = x2_o[idx[1] + 665]

        if args.task_type == 'LMI':
            # dataset1
            entity1 = x2_o[idx[0]]
            entity2 = x2_o[idx[1] + 645]

        # multi-relationship modelling decoder
        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)

        feature = torch.cat((add, product, concatenate), dim=1)

        log1 = F.relu(self.decoder1(feature))
        log = self.decoder2(log1)

        return log, ret_os, ret_os_a, x2_o, logits, log1


