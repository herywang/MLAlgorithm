import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_len = 32
seq_len = 5
batch_size = 16


class SelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, x):
        '''
        @param: x shape(batch_size, seq_len, embedding_size)
        '''
        assert len(x.shape) == 3, '输入句子向量维度不正确!'
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        t1 = torch.bmm(queries, keys.transpose(1, 2))
        t2 = F.softmax(t1 / torch.sqrt(torch.tensor(self.embedding_size)),dim=2)
        attention = torch.bmm(t2, values)
        return attention
        # return torch.bmm(F.softmax(torch.bmm() / torch.sqrt(torch.tensor(self.embedding_size)),dim=1), values)


if __name__ == '__main__':
    # 用随机模拟生成一个批次为16, 句子长度为5, embedding_size为32的输入数据
    x = torch.randn(batch_size, seq_len, embedding_len)
    model = SelfAttention(embedding_len)
    output = model(x)
    print(output.shape)  


class SelfAttention1:
    def __init__(self):
        # 用随机模拟生成一个批次为16, 句子长度为5, embedding_size为32的输入数据
        x = torch.randn(16, 5, 32)
        # 计算Attention权重W
        W = torch.bmm(x, x.transpose(1, 2)) / torch.sqrt(torch.tensor(32))
        # 加上softmax
        print(W.shape)
        W = F.softmax(W, dim=2)
        y = torch.bmm(W, x)
        print(y.shape)
        # print(x)
        # print("===")
        # print(W)
        a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        b = F.softmax(a, dim=0)
        print(b)


class SelfAttention2(nn.Module):
    def __init__(self):
        super(SelfAttention2, self).__init__()

        self.tokeys = nn.Linear(embedding_len, embedding_len)
        self.toqueries = nn.Linear(embedding_len, embedding_len)
        self.tovalues = nn.Linear(embedding_len, embedding_len)

    def forward(self, x):
        '''
        :param x: shape=(B, n, embedding_len)
        :return:
        '''
        b, n, e = x.size()
        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)
        '''
        b: batch size, 16
        n: sentence length, 5
        e: embedding size, 32
        keys: (16, 5, 32)
        queries: (16, 5, 32)
        values: (16,5,32)
        w: (16, 5,5)
        '''
        w = torch.bmm(keys, queries.transpose(1, 2))
        w = w / torch.sqrt(e)
        w = F.softmax(w, dim=2)

        out = torch.bmm(w, values)
        # out shape: (16, 5, 32)

        print(keys.shape, queries.shape, values.shape)
        return out
