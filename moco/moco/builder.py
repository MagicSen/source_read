# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # resnet最后一个fc层输出的特征维数为128
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # 如果启用mlp层，那么输出前需要再加一层relu与fc
        # 增加projection header
        if mlp:  # hack: brute-force replacement
            # 新加的fc层与之前fc层输入的尺寸一致
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        # k 复制 q的结果
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # 注册字典队列到self._buffer字典中不会被优化器更新，需要自动更新，大小为dim X K, 这里是 128 * 65536
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # 注册队列指针
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # 动量方式更新encoder参数
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # encoder_k的参数依靠encoder_q的参数来 动量式的更新
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        ## 获取分布式训练的keys
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        # 得到batch_size
        batch_size = keys.shape[0]
        # 得到队列指针
        ptr = int(self.queue_ptr)
        # 设置的字典长度必须整除 batch_size
        assert self.K % batch_size == 0  # for simplicity
        # 更新队列，将队列当前 前ptr ~ ptr+batch_size行替换为当前行
        # 假设batch_size为128，一共4个gpu，则每次替换512行，约128轮，队列完全替换完一次
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        # 循环更新队列下标
        ptr = (ptr + batch_size) % self.K  # move pointer
        # 记录队列下标
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # 只支持数据并行
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # 获取分布式训练所有节点的x，并且连接到一起
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        # 得到数据总数
        batch_size_all = x_gather.shape[0]
        # 得到GPU数目
        num_gpus = batch_size_all // batch_size_this

        # 获取随机下标，batch_size_all为全部节点一个batch的总数
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # 随机下标从0号gpu广播到全体
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # 获得随机数排序后下标，可以用来反推出排序结果
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # 获得gpu编号
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        # 按number_gpu划分，随机list
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        # 获得每个gpu相关
        return x_gather[idx_this], idx_unshuffle

    # 反算出随机的batch
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # 第一步计算query经过encoder后的feature
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        # L2范数归一化
        q = nn.functional.normalize(q, dim=1)

        # 字典分支，需要根据momentum方式更新参数
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # 基于query分支的encoder，采用动量方式更新字典分支的encoder
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle BN 避免小batch内部信息泄露
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # key经过字典分支的encoder计算feature
            k = self.encoder_k(im_k)  # keys: NxC
            # L2范数归一化
            k = nn.functional.normalize(k, dim=1)
            # 反算shuffle
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # 计算正例的loss，采用爱因斯坦简记法，按行点积求和
        # 1个正例
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # 出队列，求两个矩阵相乘
        # K个负例
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # 将正例与负例的loss求和
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # 除以中间变量
        # apply temperature
        logits /= self.T
        # 求得softmax label
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # 更新队列
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

# 收集分布式训练后的参数
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # 构建进程数个与tensor同尺寸的全一张量
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    # 保存分布式的tensor到tensors_gather
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    # 单机多cpu的tensor结果整合
    output = torch.cat(tensors_gather, dim=0)
    return output
