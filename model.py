import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SANe(torch.nn.Module):
    def __init__(self, params):
        super(SANe, self).__init__()
        self.p = params
        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim, padding_idx=None)
        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2, self.p.embed_dim, padding_idx=None)
        self.year_embed = torch.nn.Embedding(self.p.n_year, self.p.embed_dim, padding_idx=None)
        self.month_embed = torch.nn.Embedding(self.p.n_month, self.p.embed_dim, padding_idx=None)
        self.day_embed = torch.nn.Embedding(self.p.n_day, self.p.embed_dim, padding_idx=None)

        self.register_parameter('bias', nn.Parameter(torch.zeros(self.p.num_ent)))

        self.bceloss = torch.nn.BCELoss()

        self.chequer_perm = self.p.chequer_perm

        # ----
        self.inp_drop = nn.Dropout(self.p.inp_drop)
        self.feature_drop_l1 = nn.Dropout(self.p.feat_drop)
        self.feature_drop_l2 = nn.Dropout(self.p.feat_drop)
        self.feature_drop_l3 = nn.Dropout(self.p.feat_drop)
        self.hidden_drop = nn.Dropout(self.p.hid_drop)
        self.num_filt = [self.p.num_filt, self.p.num_filt, self.p.num_filt]
        self.ker_sz = [self.p.ker_sz, self.p.ker_sz, self.p.ker_sz]

        flat_sz_h = self.p.k_h
        flat_sz_w = 2 * self.p.k_w
        self.padding = 0
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt[-1]

        self.bnl0 = nn.BatchNorm2d(1)
        self.bnl2 = nn.BatchNorm2d(self.num_filt[1])
        self.bnl3 = nn.BatchNorm2d(self.num_filt[2])
        self.bnl1 = nn.BatchNorm2d(self.num_filt[0])
        self.bnfn = nn.BatchNorm1d(self.p.embed_dim)
        # ----
        self.param_gene_l1 = nn.Linear(self.p.embed_dim, self.num_filt[0] * 1 * self.ker_sz[0] * self.ker_sz[0])
        self.param_gene_l2 = nn.Linear(self.p.embed_dim,
                                       self.num_filt[1] * self.num_filt[0] * self.ker_sz[1] * self.ker_sz[1])
        self.param_gene_l3 = nn.Linear(self.p.embed_dim,
                                       self.num_filt[2] * self.num_filt[1] * self.ker_sz[2] * self.ker_sz[2])

        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

        self.time_encoder = torch.nn.RNN(input_size=self.p.embed_dim, hidden_size=self.p.embed_dim, bidirectional=False,
                                         batch_first=True)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.ent_embed.weight)
        nn.init.xavier_normal_(self.rel_embed.weight)
        nn.init.xavier_normal_(self.year_embed.weight)
        nn.init.xavier_normal_(self.month_embed.weight)
        nn.init.xavier_normal_(self.day_embed.weight)

    def forward(self, sub, rel, year, month, day, neg_ents, strategy='one_to_x'):
        h_emb = self.ent_embed(sub)
        r_emb = self.rel_embed(rel)
        y_emb = self.year_embed(year)
        m_emb = self.month_embed(month)
        d_emb = self.day_embed(day)

        time_emb = self.time_encoder(torch.stack([y_emb, m_emb, d_emb], 1))[0]
        y_emb = time_emb[:, 0, :]
        m_emb = time_emb[:, 1, :]
        d_emb = time_emb[:, 2, :]

        y_emb = self.param_gene_l1(y_emb).view(-1, self.num_filt[0], 1, self.ker_sz[0], self.ker_sz[0])

        comb_emb = torch.cat([h_emb, r_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        stack_inp = self.bnl0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = self.circular_padding_chw(x, self.ker_sz[0] // 2)
        # ------------------------------
        Batch, FN, C, FH, FW = y_emb.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * 0 - FH) // 1)
        out_w = int(1 + (W + 2 * 0 - FW) // 1)
        x = F.unfold(x, (self.ker_sz[0], self.ker_sz[0]))
        x = torch.bmm(x.transpose(1, 2), y_emb.view(Batch, y_emb.size(1), -1).transpose(1, 2)).transpose(1, 2)
        x = F.fold(x, (out_h, out_w), (1, 1))
        # ------------------------------
        x = self.bnl1(x)
        x = torch.relu(x)

        x = self.feature_drop_l1(x)
        x = self.circular_padding_chw(x, self.ker_sz[1] // 2)

        m_emb = self.param_gene_l2(m_emb).view(-1, self.num_filt[1], self.num_filt[0], self.ker_sz[1], self.ker_sz[1])
        Batch, FN, C, FH, FW = m_emb.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * 0 - FH) // 1)
        out_w = int(1 + (W + 2 * 0 - FW) // 1)
        x = F.unfold(x, (self.ker_sz[1], self.ker_sz[1]))
        x = torch.bmm(x.transpose(1, 2), m_emb.view(Batch, m_emb.size(1), -1).transpose(1, 2)).transpose(1, 2)
        x = F.fold(x, (out_h, out_w), (1, 1))
        x = self.bnl2(x)
        x = torch.relu(x)

        x = self.feature_drop_l2(x)
        x = self.circular_padding_chw(x, self.ker_sz[2] // 2)

        d_emb = self.param_gene_l3(d_emb).view(-1, self.num_filt[2], self.num_filt[1], self.ker_sz[2], self.ker_sz[2])
        Batch, FN, C, FH, FW = d_emb.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * 0 - FH) // 1)
        out_w = int(1 + (W + 2 * 0 - FW) // 1)
        x = F.unfold(x, (self.ker_sz[2], self.ker_sz[2]))
        x = torch.bmm(x.transpose(1, 2), d_emb.view(Batch, d_emb.size(1), -1).transpose(1, 2)).transpose(1, 2)
        x = F.fold(x, (out_h, out_w), (1, 1))
        x = self.bnl3(x)
        x = torch.relu(x)
        x = self.feature_drop_l3(x)

        x = x.view(sub.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bnfn(x)
        x = torch.relu(x)

        if strategy == 'one_to_n':
            x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
            x += self.bias[neg_ents]

        pred = torch.sigmoid(x)
        return pred

    def loss(self, pred, true_label):
        loss = self.bceloss(pred, true_label)
        return loss

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded