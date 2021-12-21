import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embed size need to be dived by size."
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.querys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim * self.heads, self.embed_size)

    def forward(self, values:torch.Tensor, keys:torch.Tensor, queries:torch.Tensor, mask):
        '''
        :param values: shape=(batches, sentence_length, embedding_dim)
        :param keys:
        :param queries:
        :param mask:
        :return:
        '''
        # 训练多少个样本(一批有多少个数据)
        N = queries.shape[0]
        # 语句的长度
        # 输入shape:(batches, sentence_length, word_dim(excetly embed_dim))
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # 多头注意力机制, 将embedding维度分隔成 heads * head_dim
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.head, self.head_dim)
        queries = queries.reshape(N, query_len, self.head, self.head_dim)

        '''
        queries shape: (N, query_len, heads, head_dim)
        values shape: (N, value_len, heads, head_dim)
        keys shape: (N, key_len, heads, head_dim)
        
        --> energy shape: (N,heads, query_len, key_len)
        '''
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            # 将mask矩阵中0的位置填充为-1e20, mask的作用是模拟出RNN的效果:即
            energy = energy.masked_fill(mask==0, float("-1e20"))



class BaseSelfAttention:
    def __init__(self,embed_size):
        super(BaseSelfAttention,self).__init__()

    def forward(self, queries, keys, values):
        queries @ torch.transpose()
