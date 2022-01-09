import argparse

import einops
from einops import rearrange
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pair(t: [tuple or int]):
    '''
    获得图像的宽,高. 如果传递一个数, 则表示图像宽高均为这个数
    :param t:
    :return:
    '''
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        '''
        :param dim: embedding tensor dimension
        :param heads: number of heads in multi-head attention
        :param dim_head: head dimmension
        :param dropout: drop out probability
        '''
        super(Attention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3)

        is_multi_head = not (heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if is_multi_head else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = rearrange(qkv[0], 'b n (h d)-> b h n d', h=self.heads), \
                  rearrange(qkv[1], 'b n (h d)-> b h n d', h=self.heads), \
                  rearrange(qkv[2], 'b n (h d)-> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.dim_head ** -0.5
        attn = nn.Softmax(-1)(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        '''
        :param dim: 线性映射层输出的维度(相当低于transformer网络输入的维度)
        :param depth: transformer block的数量
        :param heads: number of head
        :param dim_head: head dimention
        :param mlp_dim: feed forward layer output dimension
        :param dropout: feed forward drop out比率
        '''
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Attention Module
                nn.Sequential(
                    nn.LayerNorm(dim),  # 第一层: layer normalization
                    Attention(dim, heads, dim_head, dropout),  # Secondary layer: attention layer
                    nn.LayerNorm(dim),  # Third layer: layer normalization
                ),
                # Feed Forward Module
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差操作
            x = ff(x) + x
        return x


class ViTNet(nn.Module):
    def __init__(self, image_size, patch_size, num_class, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, channels=3, dropout=0., emb_dropout=0.):
        super(ViTNet, self).__init__()
        # 获得图像的高, 宽
        image_height, image_width = pair(image_size)
        # 获得patch的维度, 一个图像被分成了多少份patch
        patch_height, patch_width = pair(patch_size)
        # 对patch进行验证
        assert image_height % patch_height == 0 and image_width % patch_width == 0, '图像的维度必须能够被patch的维度整除'
        num_patches = (image_width // patch_width) * (image_height // patch_height)
        # 计算每个patch展开后的维度
        patch_dim = channels * patch_height * patch_width

        # -------------开始定义pytorch网络操作-------------
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_tocken = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        # 最后一层分类输出
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_class)
        )

    def forward(self, image):
        x = self.to_patch_embedding(image)
        b, n, d = x.shape
        # 重复复制batch个cls_tocken
        cls_tockens = einops.repeat(self.cls_tocken, '() n d -> b n d', b=b)
        x = torch.cat([cls_tockens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        # 经过transformer层计算后, 输出的向量是一个三维的,(batch_size, seq_len, embedding_size),
        # 这时我们需要使用某种策略将其缩减为二维, 符合全连接层的标准输入规范(batch_size, vector_size),
        # 因为attention中我们将图像分成了seq_len个patch, 那么将这些组合一起求个均值是否能抽象成为这张图片
        # 的表征? 因此通常会使用mean的方式, 当然也有使用其它的方式, 例如: cls, 去第一个patch作为表征向量
        # 传入到全连接层. 不过这都无所谓, 都可以被优化算法进行训练
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


def main():
    parser = argparse.ArgumentParser(description="Pytorch Vision Transformer Example")
    parser.add_argument("--batch-size", type=int, default=32, help="输入训练的batch-size大小")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="输入测试的batch-size大小")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.1307,))
    ])
    args = parser.parse_args()
    train_args = {"batch_size": args.batch_size}
    test_args = {"batch_size": args.test_batch_size}

    # 下载训练数据集
    train_dataset = datasets.CIFAR100("./data", True, transform, download=True)
    # 下载测试数据集
    test_dataset = datasets.CIFAR100("./data", False, transform, download=True)

    # 构建Train loader和Test loader
    train_loader = DataLoader(train_dataset, **train_args)
    test_loader = DataLoader(test_dataset, **test_args)

    model = ViTNet(28, 4, 10, 128, 5, 1, 128, 'mean', dim_head=64, channels=1)

    # ========以下代码写自己的优化函数及训练过程==========



if __name__ == '__main__':
    model = ViTNet(28, 4, 10, 128, 5, 7, 128, 'mean', dim_head=64, channels=1)
    image = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        pred = model(image)
    print(pred.shape)
    print(pred)
