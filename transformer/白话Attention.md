# Transformer从入门到入土

**Transformer是如今机器学习领域一个非常重要的网络架构, 这篇文章我们就来详细介绍Transformer是怎么工作的. 由于Transformer最初是用来解决NLP领域的问题, 因此本文我们也分三个部分来学习Transformer:**  

1. 循环神经网络系列(以RNN举例).  
2. Self-Attention.  
3. Transformer.

## 1. RNN翻译任务举例

&emsp;&emsp;假设有一句话 `你好吗`, 现在我们需要将其翻译成英文: `How ary you`, 使用RNN神经网络的结构通常是这样的:  
<div align="center">
<img src="./images/1.png" alt="image-20211221135825429" style="zoom:40%;align=center;" />
</div>

&emsp;&emsp;由于循环神经网络的特性, 训练过程不能并行, 网络首先将`你`传递给隐藏神经元,翻译得到`How`,然后将上一次计算的隐藏层数据$h_1$和`好`一起传递给隐藏层,计算得到`Are`, 重复上述操作, 最后得到翻译的`You`. 显然这样需要进行三次循环才能完成一个句子(Sentence)的翻译任务. 在训练过程中有大量的训练sentence.  

&emsp;&emsp;在RNN计算过程中需要彻底弄清楚输入向量和输出向量的shape, 这里先了解几个概念:  

* **词典:** 场景中全部词汇, 例如:维基百科中出现的全部词汇(英文词典), 百度百科出现的全部汉字(汉语词典). 通常情况下英文中一个单词为一个"词", 汉语中一个字为一个"词"  

* **词典长度:** (vocabulary length) 词典中所有"词"的数量. 例如世界上常用的全部英文大约有2万个,那么词典长度voca_len=20000, 中文中常用的汉字有2000个, 中文词典长度voca_len=2000.  

* **词向量:** 针对每一个词我们怎么用向量的方式将其表达出来, 一中方式是`one-hot`方式, 即:假设一个词$a$在词典中的位置是第987个, 那么这个词用就可以标识为:
    $$
    a = [0,0,0,...,1,...,0,0,0]
    $$
    即: 第987位是1, 其它位全是0.  
    上面这种方式显然空间浪费非常大, 因此通常我们使用`Word Embedding(词嵌入)`的方式, 将一个词嵌入到低纬空间中:
    <div align="center">
    <img src="./images/2.png" alt="image-20211221150149138" style="zoom:40%;" />
    </div>  

    针对一个词向量
    $$x_1 = [0,0,0,...,1,0,...,0], shape:(1, voc\_len)$$
    初始化一个权重矩阵:
    $$W, shape:(voc\_len, embed\_size)$$  
    经过Embedding层计算最终得到输出:  
    $$ output = x_1 × W, shape:(1, embed\_size) $$  
    由此可见, 针对一个长度为n句子(sentence), 用one-hot矩阵表示形式为:
    $$
    X = \left[ \begin{matrix} 
    0 & 1 & 0 & ... & 0\\
    1 & 0 & 0 & ... & 0\\
    0 & 0 & 1 & ... & 0 \\
    ... & ... & ... & ... & ...\\
    0 & 0 & 0 & ... & 1 
    \end{matrix} \right],shape:(n, voc\_len)
    $$
    输入$X$到Embedding层, 最终得到一个shape为$(n, embed_size)$的向量. 仔细观察$output = X * W$就能发现, 一个由one-hot向量构成的矩阵乘以权重矩阵$W$得到的结果其实就是矩阵$W$对应one-hot向量中`1`元素的索引. 我们举个🌰看看:   
    假设:
    $$
    X = \left[ \begin{matrix} 
    0 & 1 & 0 & 0 & 0\\
    1 & 0 & 0 & 0 & 0\\
    0 & 0 & 1 & 0 & 0\\
    0 & 0 & 0 & 1 & 0
    \end{matrix} \right], 
    W = \left[ \begin{matrix} 
    w_{11} & w_{12} & w_{13}\\
    w_{21} & w_{22} & w_{23}\\
    w_{31} & w_{32} & w_{33}\\
    w_{41} & w_{42} & w_{43}\\
    w_{51} & w_{52} & w_{53}
    \end{matrix} \right]
    $$
    那么:
    $$
    output = X×W = \left[ \begin{matrix} 
    0 & 1 & 0 & 0 & 0\\
    1 & 0 & 0 & 0 & 0\\
    0 & 0 & 1 & 0 & 0\\
    0 & 0 & 0 & 1 & 0
    \end{matrix} \right] × \left[ \begin{matrix}
    w_{11} & w_{12} & w_{13}\\
    w_{21} & w_{22} & w_{23}\\
    w_{31} & w_{32} & w_{33}\\
    w_{41} & w_{42} & w_{43}\\
    w_{51} & w_{52} & w_{53}
    \end{matrix} \right] = \left[ \begin{matrix}
    w_{21} & w_{22} & w_{23}\\
    w_{11} & w_{12} & w_{13}\\
    w_{31} & w_{32} & w_{33}\\
    w_{41} & w_{42} & w_{43}
    \end{matrix} \right]
    $$
    由此可见, 我们其实没必要将$x_i$转换成one-hot编码形式, 只需要输入当前$x_i$所在词典中的索引即可.  

----
**有了上述基础知识之后, 下面我们需要思考一个问题: RNN由于其循环神经网络的特性, 每个句子训练学习过程中只能一个词计算完毕之后,再计算另一个词. 那么有没有一种方式能让网络同时模拟出RNN的<font color="red">上文相关性</font>, 又能并行训练学习? Attention机制其实干的就是这件事**  

## 2. Attention机制详解  

还是针对上述的翻译案例, 我们先看一幅图:
<div style="float:center;">
    <img src="./images/3.png" alt="image-20211221160721597" style="zoom: 33%;" />
    <img src="./images/4.png" alt="image-20211221160958509" style="zoom:33%;border-left:5px solid red; margin-left:30px;" />
</div>

计算过程如下:
$$
y_1 = x_1 * w_{11} + x_2 * w_{12} + x_3 * w_{13} \\
y_2 = x_1 * w_{21} + x_2 * w_{22} + x_3 * w_{23} \\
y_3 = x_1 * w_{31} + x_2 * w_{32} + x_3 * w_{33} \tag{2.1}
$$
这时就能够做到并行计算, 其中参与计算的权重$w_{ij}$我们称之为: Attention值, 是一个标量, 说白了Attention就是做了一个权重再分配, 说人话就是: 计算$y_1$时,我到底应该更关注哪个输入?  

明白了Attention机制后, 接下来就需要确定Attention值是怎么计算得到的, 即:权重矩阵
$$
W = \left[ \begin{matrix} 
    w_{11} & w_{12} & ... & w_{1m}\\
    w_{21} & w_{22} & ... & w_{2m}\\
    w_{31} & w_{32} & ... & w_{3m}\\
    ... & ... & ... &...\\
    w_{m1} & w_{m2} & ... & w_{mm}
    \end{matrix} \right], 其中m表示序列(sequence)的长度.\tag{2.2}
$$
值是怎么计算得到的. 这时我们回到经典论文: **[Attention is All you Need](https://arxiv.org/abs/1706.03762)**, 论文中定义的Attention计算公式为:
$$
Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V \tag{2.3}
$$
这就好办了, 回到上面我绘制的那张图:

<img src="./images/5.png" alt="image-20211221164210311" style="zoom:67%;" />

怎么计算Attention的值这张图应该一眼就能够看出来吧? 在计算$w_{11}$时,横着的$x_1$为Query, 竖着的$x_1$表示Key, 计算$w_{12}$时,横着的$x_1$为Query, 竖着的$x_2$表示Key, 计算$w_{13}$时,横着的$x_1$为Query, 竖着的$x_3$表示Key, 那么$w_{11}, w_{12},w_{13}$结合Attention计算公式的具体计算表达式如下所示:  
$$
\frac{e^{\frac{x_1{x_j}^T}{\sqrt{4}}}}{\sum_{j}{e^{\frac{x_1{x_j}^T}{\sqrt{4}}}}} * x_j \tag{2.4}
$$
公式中左边那一大坨就是$Softmax(QK^T)$, $x_j$就是$V$.  

## 3. 为什么Attention机制能够正常工作?
这一节我们主要来解释一下为什么Attention能够work. 假设在一个电影推荐的任务上,对一个电影归类我们假设有四个维度:`[三级片段, 动作片段, 喜剧片段, 浪漫片段]`, 对用户喜欢某种电影我们也设置四个维度:`[喜欢三级片, 喜欢动作片, 喜欢喜剧片, 喜欢浪漫片]`, 看下面一张图, 如何衡量用户$u$是否喜欢电影$m$? 
<img src="./images/6.png" alt="image-20211221171242668" style="zoom:35%;" />  
用户特征向量和电影特征向量相乘难道不就是我们熟悉的向量近似度计算公式? 如果向量$u$和向量$m$乘积越大, 说明这两个向量近似度越高, 否则说明这两个向量近似度越低.

再仔细观察一下权重$w$的计算公式:
$$
w_{ij} = Softmax(\frac{x_i{x_j}^T}{\sqrt{d}}) \tag{3.1}
$$

在计算Attention值时, 我们是否可以认为计算过程中关注那些相关度较高的词参与翻译的推理计算?  
例如: 给定一个句子:
$$
v_{the}, v_{cat}, v_{walks}, v_{on}, v_{the}, v_{street}
$$
我们要将其翻译成:
$$
y_{猫}, y_{走}, y_{在}, y_{街}, y_{上}
$$
在翻译$y_猫$的时候, 通过计算发现$v_{cat}$的权重最高,$v_{the}以及其他的权重较低$, 因此注意力有90\%集中在计算$v_{cat}$, 其他的权重就分配的较低.

有了上述2, 3节的知识点, 下面我们实现一个小小的Self-Attention的例子:

```python
import torch
import torch.nn.functional as F
# 用随机模拟生成一个批次为16, 句子长度为5, embedding_size为32的输入数据
x = torch.randn(16, 5, 32)
# 计算Attention权重W
W = torch.bmm(x, x.transpose(1,2))
# 加上softmax
W = F.softmax(W, dim=2)
# 输出到y
y = torch.bmm(W, x)
print(y)
```

但是在Transformer中使用Attention并不是这么直接简单的将向量$X$作为keys, queries, values参与计算, 而是使用了一个小trick将其与一个权重相乘得到一个向量, 即: 定义:
$$
q_i = W_qx_i\\
k_i = W_kx_i\\
v_i = W_vx_i\\
$$
此时计算$w_{1j}$计算就变成了:
$$
w_{1j} = \frac{e^{\frac{(W_qx_1)^T(W_kx_j)}{\sqrt{4}}}}{\sum_{j}{e^{\frac{(W_qx_1)^T(W_kx_j)}{\sqrt{4}}}}} \tag{3.2}
$$
其中$W_q, W_k, W_v$是随机初始的权重, 后期被训练优化.
下面我们再实现一下经过调整后的SelfAttention:

```python
embedding_len = 32
seq_len = 5
batch_size = 16
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
        w = torch.bmm(keys, queries.transpose(1,2))
        w = w / torch.sqrt(e)
        w = F.softmax(w, dim=2)

        out = torch.bmm(w, values)
        # out shape: (16, 5, 32)

        print(keys.shape, queries.shape, values.shape)
        return out

if __name__ == '__main__':
    # 用随机模拟生成一个批次为16, 句子长度为5, embedding_size为32的输入数据
    x = torch.randn(batch_size, seq_len, embedding_len)
    model = SelfAttention2()
    model(x)
```

## 4. 多头注意力机制

![image-20220102135736196](./images/9.png)

上图是两个翻译任务:  
①你给我钱->You give me money.
②我给你钱->I give you money.
如果使用普通的Attention计算就会存在一定的问题: 当我们计算第二个单词`给`的时候, 两句话的attention是一样的. 但是这两句话所表达的意思完全不一样, 需要做一个区分. 这时就引入了多头attention机制, 多头attention机制的实现是使用一个线性层,将输入的向量映射到一个$h * head_dim$的空间中, 其中h表示"头"的数量, "head_dim"是头的维度. Transformer论文中使用的是8个头.  

**举个简单的🌰:** 假如一个词我们将其嵌入到32维度的向量中, 头的数量我们设置成8个, 那么这个词的每4个维度的向量为1组, 分成8组分别传递到8个"头"中进行计算, 就相当于把一个词截断成8节, 然后在进行计算Attention值.  分别计算完毕之后在将8个头计算的结果concat到一起, 然后传递到一个线性层中, 映射得到和输入向量一模一样的词向量. 如下图所示:  

<img src="./images/7.png" alt="image-20211221175006156" style="zoom:53%;" />

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        # 获取batch size
        N = query.shape[0]
        # 获取句子(序列sequence)长度
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 做一下维度的切分, 满足多头注意力机制的输入shape
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # 使用Einsum库做矩阵想乘, 多头注意力搞矩阵转置, 乘法可直接使用这个库进行, 避免了我们手动写转置然后再想乘可能出现的一些错误, 如果对einsum有疑问可参考pytorch的一些官方文档.
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim)
        # 然后将其reshape成和输入一模一样的向量, 最终再使用一个全连接成进行映射一下.

        out = self.fc_out(out)
        # (N, query_len, embed_size)
        return out
```

# 5. 手撕Transformer

Transformer结构:  

<img src="./images/8.png" alt="image-20211221193936847" style="zoom:63%;" />

理解Transformer前我们先弄清楚Transformer网络的输入和输出, 由上面Transformer网络模型结构可以看到, 网络分为Encoder模块和Decoder模块. 假如有一个机器翻译任务是: 将"我爱机器学习" 翻译成 "I Love machine learning", 其训练过程是:  

> 1. 首先将"我爱机器学习"输入到Encoder层, 对这句话进行编码, 计算获得一个输出output, 将这个输出作用与每个decoder的K和V
> 2. decoder层输入开始符<bos>, 将decoder的最大概率输出词$A_1$和"i"做cross entropy计算loss
> 3. decoder层输入<bos>, "i", 将decoder的最大概率输出词$A_2$和"love"做cross entropy计算loss
> 4. decoder层输入<bos>, "i", "love", 将decoder的最大概率输出词$A_3$和"machine"做cross entropy计算loss
> 5. decoder层输入<bos>, "i", "love","machine",将decoder的最大概率输出词$A_4$和"learning"做cross entropy计算loss
> 6. decoder层输入<bos>, "i", "love","machine","learning",将decoder的最大概率输出词$A_5$和终止符</s>做cross entropy计算loss

由此可见上述训练过程是串行的, 那么能不能并行呢? 答案是当然可以. 我们分析一下上述输入到decoder中的数据分别是:   
> \<bos>   
> \<bos>, "i"  
> \<bos>, "i", "love"  
> \<bos>, "i", "love","machine"  
> \<bos>, "i", "love","machine","learning"

我们可以将其整成一个矩阵的形式, 并在其前面乘以一个掩码就能得到相应的输出:
$$
    O = \left[ \begin{matrix} 
    1 & 0 & 0 & 0 & 0\\
    1 & 1 & 0 & 0 & 0\\
    1 & 1 & 1 & 0 & 0\\
    1 & 1 & 1 & 1 & 0\\
    1 & 1 & 1 & 1 & 1\\
    \end{matrix} \right] × \left[ \begin{matrix} 
    <bos> & i & love & machine & learning\\
    <bos> & i & love & machine & learning\\
    <bos> & i & love & machine & learning\\
    <bos> & i & love & machine & learning\\
    <bos> & i & love & machine & learning
    \end{matrix} \right]  \\
    = \left[ \begin{matrix} 
    <bos> \\
    <bos> & i \\
    <bos> & i & love \\
    <bos> & i & love & machine \\
    <bos> & i & love & machine & learning
    \end{matrix} \right]
$$
Transformer在预测的时候, 首先将需要翻译的句子送到Encoder模块, 然后在decoder层输入开始符号[\<bos>], 然后输出预测下最大概率的下一个词$A_1$,接着将[\<bos\>,"$A_1"$]输入到decoder, 得到下一个预测词$A_2$, 重复以上步骤, 直到输出结束符为止, 就完成了一轮预测. 