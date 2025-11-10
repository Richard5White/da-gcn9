import os.path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DataSet
from utils import BPRLoss, EmbLoss, ContrastiveLoss, init_embedding


#这是论文源代码
class SpAdjEdgeDrop(nn.Module):
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edge_num = vals.size()
        mask = (torch.rand(edge_num) + keep_rate).floor().type(torch.bool)
        new_vals = vals[mask]
        new_idxs = idxs[:, mask]
        return torch.sparse.FloatTensor(new_idxs, new_vals, adj.shape)


# 在model.py中修改GCN类，增加动态邻接矩阵调整
class GCN(nn.Module):
    def __init__(self, layers, args):
        super(GCN, self).__init__()
        self.layers = layers
        self.edge_dropper = SpAdjEdgeDrop()
        self.embedding_size = args.embedding_size
        self.gcn_method = args.gcn_method
        self.if_add_weight, self.if_add_bias = args.if_add_weight, args.if_add_bias


        if self.if_add_weight:
            self.linear_layers = nn.ModuleList(
                [nn.Linear(self.embedding_size, self.embedding_size, self.if_add_bias) for _ in range(self.layers)]
            )

        # 新增：动态邻接矩阵权重调整模块
        self.adj_weight = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.ReLU(),
            # 增加元素积特征交互
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, adj, keep_rate):
        all_embeddings = [x]
        # 分离用户和物品嵌入用于计算边权重
        num_users = x.shape[0] - (adj.shape[0] - x.shape[0])  # 假设x是[用户嵌入; 物品嵌入]
        self.user_attention = nn.Embedding(num_users + 1, 3)
        user_emb = x[:num_users]
        item_emb = x[num_users:]

        for i in range(self.layers):
            _adj = self.edge_dropper(adj, keep_rate)

            # 动态计算边权重：基于节点嵌入相似度
            if self.training:  # 仅训练时更新边权重
                edges = _adj._indices()
                u_idx = edges[0]  # 用户节点索引
                v_idx = edges[1]  # 物品节点索引

                # 区分用户-物品边和物品-用户边（如果有双向边）
                is_user_item = (u_idx < num_users) & (v_idx >= num_users)
                is_item_user = (u_idx >= num_users) & (v_idx < num_users)

                # 计算边权重
                edge_weights = []
                if torch.any(is_user_item):
                    u_emb = user_emb[u_idx[is_user_item] - 1]
                    i_emb = item_emb[v_idx[is_user_item] - num_users - 1]
                    # 新增元素积交互特征
                    uv_product = u_emb * i_emb
                    uv_feat = torch.cat([u_emb, i_emb, uv_product], dim=1)  # 融合拼接和元素积特征
                    edge_weights.append(self.adj_weight(uv_feat).squeeze())

                if torch.any(is_user_item):
                    u_emb = user_emb[u_idx[is_user_item] - 1]
                    i_emb = item_emb[v_idx[is_user_item] - num_users - 1]
                    uv_product = u_emb * i_emb

                    # 获取用户对不同特征的注意力权重
                    user_ids = u_idx[is_user_item]
                    att_weights = F.softmax(self.user_attention(user_ids), dim=1)  # (batch, 3)

                    # 加权融合特征
                    feat_list = [u_emb.unsqueeze(1), i_emb.unsqueeze(1), uv_product.unsqueeze(1)]
                    fused_feat = torch.cat(feat_list, dim=1)  # (batch, 3, emb_size)
                    weighted_feat = (fused_feat * att_weights.unsqueeze(-1)).sum(dim=1)  # (batch, emb_size)

                    edge_weights.append(self.adj_weight(weighted_feat).squeeze())

                # 更新邻接矩阵权重
                if edge_weights:
                    new_vals = torch.cat(edge_weights) * _adj._values()
                    _adj = torch.sparse.FloatTensor(edges, new_vals, adj.shape)

            x = torch.sparse.mm(_adj, x)
            if self.if_add_weight:
                x = self.linear_layers[i](x)
            all_embeddings.append(x)

        if self.gcn_method == 'mean':
            x = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        return x


class DAGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(DAGCN, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.checkpoint = args.checkpoint
        self.if_load_model = args.if_load_model

        # Base
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        # 添加新的嵌入优化相关参数
        self.emb_init_type = args.emb_init_type
        self.emb_reg_type = args.emb_reg_type
        self.contrast_weight = args.contrast_weight
        self.contrast_loss = ContrastiveLoss(temperature=args.contrast_temp)

        self.behaviors = dataset.behaviors
        # 新增：将'all'加入行为列表，用于初始嵌入的调整
        all_behaviors = ['all'] + self.behaviors
        # 初始化行为感知调整参数（包含'all'）
        self.behavior_attn = nn.ModuleDict({
            beh: nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size // 4),
                nn.ReLU(),
                nn.Linear(self.embedding_size // 4, self.embedding_size),
                nn.Sigmoid()  # 输出0-1之间的注意力权重
            ) for beh in all_behaviors
        })
        self.behavior_bias = nn.ParameterDict({
            beh: nn.Parameter(torch.zeros(self.embedding_size))
            for beh in all_behaviors
        })
        # Modules
        self.behaviors = dataset.behaviors
        self.keep_rate = args.keep_rate
        self.if_layer_norm = args.if_layer_norm
        self.behavior_adjs = dataset.behavior_adjs
        # 添加全局图卷积层（两轮LightGCN）
        self.global_gcn = GCN(layers=2, args=args)  # 两轮卷积
        self.global_adj = dataset.behavior_adjs.get('all', None)  # 获取全局图的邻接矩阵
        # 融合权重（可调整）
        self.fusion_weight = nn.Parameter(torch.tensor(0.01), requires_grad=False)  # 初始权重0.5

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss(reg_type=args.emb_reg_type)
        self.pre_behavior_dict = dataset.pre_behavior_dict
        self.behavior_cf_layers = dataset.behavior_cf_layers
        self.personal_trans_dict = dataset.personal_trans_dict

        self.behavior_cf = defaultdict(dict)
        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            pre_behavior_cf = nn.ModuleDict()
            for pre_beh in pre_behaviors:
                pre_behavior_cf[pre_beh] = GCN(self.behavior_cf_layers[post_beh][pre_beh], args)
            self.behavior_cf[post_beh] = pre_behavior_cf

        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            for pre_beh in pre_behaviors:
                trans_mat = self.personal_trans_dict[post_beh][pre_beh].detach()
                self.personal_trans_dict[post_beh][pre_beh] = nn.Parameter(trans_mat, requires_grad=False).to(
                    self.device)

        self.behavior_weight = nn.ModuleDict({
            post_beh: nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.Tanh(),
                nn.Linear(self.embedding_size, 1),
                nn.Sigmoid()  # 输出0-1之间的权重
            ) for post_beh in self.behaviors
        })

        # Loss
        self.reg_weight = args.reg_weight
        self.aux_weight = args.aux_weight
        self.if_multi_tasks = args.if_multi_tasks
        self.mtl_type = args.mtl_type
        self.personal_loss_ratios = dataset.personal_loss_ratios
        self.global_loss_ratios = dataset.global_loss_ratios
        self.loss_ratio_type = args.loss_ratio_type

        self.storage_all_embeddings = None

        self._init_weights()
        self._load_model()

    def _init_weights(self):
        # 使用改进的嵌入初始化方法
        self.user_embedding = init_embedding(
            self.user_embedding,
            init_type=self.emb_init_type
        )
        self.item_embedding = init_embedding(
            self.item_embedding,
            init_type=self.emb_init_type
        )

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.checkpoint, 'model.pth'))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self, with_augmentation=False):
        all_embeddings = {}
        last_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings['all'] = last_embedding

        # 用于对比学习的增强视图
        aug_embeddings = {}
        if with_augmentation:
            # 通过dropout创建嵌入的不同视图，确保随机掩码在当前设备上生成
            drop_rate = 0.2
            # 关键修复：使用设备上的随机数生成器
            mask = torch.rand(last_embedding.shape[0], 1, device=self.device) > drop_rate
            aug_last_embedding = last_embedding * mask.float()
            aug_embeddings['all'] = aug_last_embedding

        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            pre_behaviors = pre_behaviors[::-1]
            post_embeddings = []

            # 增强视图的传播
            aug_post_embeddings = [] if with_augmentation else None

            for pre_beh in pre_behaviors:
                # 获取前置行为的嵌入并应用行为感知调整
                pre_embedding = all_embeddings[pre_beh].to(self.device)  # 确保设备一致
                # 核心优化：根据前置行为类型动态调整嵌入
                attn_weights = self.behavior_attn[pre_beh](pre_embedding)
                scaled_pre_emb = pre_embedding * attn_weights + self.behavior_bias[pre_beh]

                # GCN传播（使用调整后的嵌入）
                layer_adj = self.behavior_adjs[post_beh].to(self.device)  # 确保邻接矩阵在设备上
                lightgcn_emb = self.behavior_cf[post_beh][pre_beh](scaled_pre_emb, layer_adj, self.keep_rate)
                trans_mat = self.personal_trans_dict[post_beh][pre_beh].to(self.device)  # 确保转换矩阵在设备上
                post_embedding = torch.mul(trans_mat, lightgcn_emb)
                post_embeddings.append(post_embedding)

                # 增强视图的传播
                if with_augmentation:
                    aug_pre_embedding = aug_embeddings[pre_beh].to(self.device)
                    aug_lightgcn_emb = self.behavior_cf[post_beh][pre_beh](
                        aug_pre_embedding, layer_adj, self.keep_rate * 0.8)
                    aug_post_emb = torch.mul(trans_mat, aug_lightgcn_emb)
                    aug_post_embeddings.append(aug_post_emb)

            agg_messages = sum(post_embeddings)
            if self.if_layer_norm:
                agg_messages = F.normalize(agg_messages, dim=-1)
            cur_embedding = agg_messages + last_embedding
            all_embeddings[post_beh] = cur_embedding
            last_embedding = cur_embedding

            # 处理增强视图
            if with_augmentation:
                aug_agg_messages = sum(aug_post_embeddings)
                if self.if_layer_norm:
                    aug_agg_messages = F.normalize(aug_agg_messages, dim=-1)
                aug_cur_embedding = aug_agg_messages + aug_last_embedding
                aug_embeddings[post_beh] = aug_cur_embedding
                aug_last_embedding = aug_cur_embedding

        if with_augmentation:
            return all_embeddings, aug_embeddings
        return all_embeddings

    def forward(self, batch_data):
        self.storage_all_embeddings = None

        all_embeddings, aug_embeddings = self.gcn_propagate(with_augmentation=True)
        total_loss = 0
        for index, behavior in enumerate(self.behaviors):
            if self.if_multi_tasks or behavior == self.behaviors[-1]:
                data = batch_data[:, index]  # (bsz,3)
                users = data[:, 0].long()  # (bsz,)
                items = data[:, 1:].long()  # (bsz, 2)
                user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior],
                                                                     [self.n_users + 1, self.n_items + 1])

                user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)  # (bsz, 2, dim)
                item_feature = item_all_embedding[items]  # (bsz, 2, dim)
                scores = torch.sum(user_feature * item_feature, dim=2)  # (bsz, 2)

                mask = torch.where(users != 0)[0]
                scores = scores[mask]

                # MTL - Personalized
                if self.mtl_type == 'personalized':
                    if behavior == self.behaviors[-1]:
                        user_loss_ratios = torch.ones_like(users).float()
                    else:
                        user_loss_ratios = self.personal_loss_ratios[behavior][users].to(self.device)
                        user_loss_ratios = self.aux_weight * user_loss_ratios
                    user_loss_ratios = user_loss_ratios[mask]
                    total_loss += (user_loss_ratios * self.bpr_loss(scores[:, 0], scores[:, 1])).mean()

                # MTL - Global
                elif self.mtl_type == 'global':
                    if behavior == self.behaviors[-1]:
                        beh_loss_ratio = 1.0
                    else:
                        beh_loss_ratio = self.aux_weight * self.global_loss_ratios[behavior]
                    total_loss += (beh_loss_ratio * self.bpr_loss(scores[:, 0], scores[:, 1])).mean()

                # Single Task or MTL-addition
                else:
                    total_loss += (self.bpr_loss(scores[:, 0], scores[:, 1])).mean()

        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight,
                                                                  self.item_embedding.weight)
        if self.contrast_weight > 0 and self.training:
            # 对用户和物品嵌入进行对比学习
            user_emb, item_emb = torch.split(all_embeddings[self.behaviors[-1]],
                                             [self.n_users + 1, self.n_items + 1])
            aug_user_emb, aug_item_emb = torch.split(aug_embeddings[self.behaviors[-1]],
                                                     [self.n_users + 1, self.n_items + 1])

            # 只对有交互的用户计算对比损失
            users_in_batch = batch_data[:, 0, 0].unique()
            valid_users = users_in_batch[users_in_batch != 0]

            # 计算用户嵌入的对比损失
            if len(valid_users) > 0:
                user_contrast_loss = self.contrast_loss(
                    user_emb[valid_users],
                    aug_user_emb[valid_users]
                )
                total_loss += self.contrast_weight * user_contrast_loss

            # 计算物品嵌入的对比损失
            items_in_batch = batch_data[:, :, 1].unique()
            valid_items = items_in_batch[items_in_batch != 0]
            if len(valid_items) > 0:
                item_contrast_loss = self.contrast_loss(
                    item_emb[valid_items],
                    aug_item_emb[valid_items]
                )
                total_loss += self.contrast_weight * item_contrast_loss

        # 改进的嵌入正则化
        if self.emb_reg_type == 'l2':
            reg_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)
        elif self.emb_reg_type == 'l1':
            reg_loss = torch.mean(torch.abs(self.user_embedding.weight)) + \
                       torch.mean(torch.abs(self.item_embedding.weight))
        elif self.emb_reg_type == 'l2+l1':
            reg_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight) + \
                       torch.mean(torch.abs(self.user_embedding.weight)) + \
                       torch.mean(torch.abs(self.item_embedding.weight))
        else:
            reg_loss = 0

        total_loss = total_loss + self.reg_weight * reg_loss
        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]],
                                                     [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]  # (test_bsz, dim)
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))  # (test_bsz, |I|)
        return scores