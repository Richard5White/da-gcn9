import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random

import torch.nn.functional as F  # 补充这行导入



# 在utils.py中添加新的嵌入初始化和对比学习相关代码
class ContrastiveLoss(nn.Module):
    """对比学习损失，增强嵌入表达能力"""

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_i, emb_j):
        """emb_i和emb_j是同一实体的两个不同视图嵌入"""
        # 归一化
        emb_i_norm = F.normalize(emb_i, dim=1)
        emb_j_norm = F.normalize(emb_j, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(emb_i_norm, emb_j_norm.t()) / self.temperature
        batch_size = emb_i.size(0)

        # 正样本是对角线元素
        mask = torch.eye(batch_size, device=emb_i.device).bool()
        positives = sim_matrix[mask].view(batch_size, 1)

        # 负样本是其他元素
        negatives = sim_matrix[~mask].view(batch_size, -1)

        # 计算损失
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=emb_i.device)
        loss = F.cross_entropy(logits, labels)
        return loss




def init_embedding(embedding, init_type='xavier_normal'):
    """多种嵌入初始化方式"""
    if init_type == 'xavier_normal':
        nn.init.xavier_normal_(embedding.weight.data)
    elif init_type == 'xavier_uniform':
        nn.init.xavier_uniform_(embedding.weight.data)
    elif init_type == 'kaiming_normal':
        nn.init.kaiming_normal_(embedding.weight.data)
    elif init_type == 'kaiming_uniform':
        nn.init.kaiming_uniform_(embedding.weight.data)
    elif init_type == 'normal':
        nn.init.normal_(embedding.weight.data, mean=0, std=0.01)
    # 填充零向量
    embedding.weight.data[0] = 0.0
    return embedding

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        return loss


class EmbLoss(nn.Module):
    """增强版嵌入正则化损失，支持多种正则化组合"""

    def __init__(self, norm=2, reg_type='l2'):
        super(EmbLoss, self).__init__()
        self.norm = norm
        self.reg_type = reg_type  # 新增：支持不同正则化类型

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            # 排除padding的0向量
            valid_emb = embedding[1:]  # 假设index=0是padding

            if self.reg_type == 'l2':
                emb_loss += torch.norm(valid_emb, p=2)
            elif self.reg_type == 'l1':
                emb_loss += torch.norm(valid_emb, p=1)
            elif self.reg_type == 'l2+l1':
                emb_loss += 0.5 * torch.norm(valid_emb, p=2) + 0.5 * torch.norm(valid_emb, p=1)
            elif self.reg_type == 'none':
                continue
            else:
                raise ValueError(f"不支持的正则化类型: {self.reg_type}")

        # 归一化处理，避免批次大小影响
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class InfoNCELoss(nn.Module):
    """
    From SSLRec models/loss.utils
    """

    def __init__(self, temp=1.0):
        super(InfoNCELoss, self).__init__()
        self.temp = temp

    def forward(self, embeds1, embeds2, all_embeds2):
        normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
        normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))

        nume_term = - (normed_embeds1 * normed_embeds2 / self.temp).sum(-1)
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / self.temp), dim=-1))
        cl_loss = (nume_term + deno_term).sum()

        return cl_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_adj(adj, method='asym'):
    if method == 'sym':
        degree = np.array(adj.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        norm_adj = d_inv_sqrt_mat.dot(adj).dot(d_inv_sqrt_mat)

    elif method == 'asym':
        degree = np.array(adj.sum(axis=-1))
        d_inv = np.reshape(np.power(degree, -1), [-1])
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv_mat = sp.diags(d_inv)
        norm_adj = d_inv_mat.dot(adj)

    elif method == 'mean':
        degree = np.array(adj.sum(axis=-1))
        d_inv = np.reshape(np.power(degree, -1), [-1])
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv_mat = sp.diags(d_inv)
        norm_adj = adj.dot(d_inv_mat)
    else:
        norm_adj = adj
    return norm_adj.tocoo()


def _get_static_hyper_adj(inc_mat: sp.dok_matrix):
    edge_count = inc_mat.shape[1]
    edge_weight = sp.diags(np.ones(edge_count))
    dv = np.array((inc_mat * edge_weight).sum(axis=1))
    de = np.array(inc_mat.sum(axis=0))

    de_inv = np.reshape(np.power(de, -1), [-1])
    de_inv[np.isinf(de_inv)] = 0.0
    de_inv_mat = sp.diags(de_inv)
    inc_mat_transpose = inc_mat.transpose()

    dv_inv_sqrt = np.reshape(np.power(dv, -0.5), [-1])
    dv_inv_sqrt[np.isinf(dv_inv_sqrt)] = 0.0
    dv_inv_sqrt_mat = sp.diags(dv_inv_sqrt)

    g = dv_inv_sqrt_mat * inc_mat * edge_weight * de_inv_mat * inc_mat_transpose * dv_inv_sqrt_mat
    return g


def sp_mat_to_torch_sp_tensor(mat):
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(idxs, vals, shape)


def dict2set(_dict):
    _set = set()
    for k, v in _dict.items():
        for _v in v:
            _set.add((k, _v))
    return _set

