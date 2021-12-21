import torch
import torch.nn.functional as F

# 用随机模拟生成一个批次为16, 句子长度为5, embedding_size为32的输入数据
x = torch.randn(16, 5, 32)
# 计算Attention权重W
W = torch.bmm(x, x.transpose(1,2))
# 加上softmax
print(W.shape)
W = F.softmax(W, dim=2)
y = torch.bmm(W, x)
print(y.shape)
# print(x)
# print("===")
# print(W)
a = torch.tensor([1,2,3,4,5],dtype=torch.float32)
b = F.softmax(a,dim=0)
print(b)