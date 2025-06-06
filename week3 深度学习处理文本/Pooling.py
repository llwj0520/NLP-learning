#coding:utf8
import torch
import torch.nn as nn

'''
pooling层的处理
'''

#pooling操作默认对于输入张量的最后一维进行
#入参5，一次将五个元素平均 代表把五维池化为一维
layer = nn.AvgPool1d(4)  
#随机生成一个维度为3x4x5的张量
#可以想象成3条,文本长度为4,向量长度为5的样本
x = torch.rand([3, 4, 5])
print(x)
print(x.shape)
#默认是对最后一个进行池化，因此在nlp任务中需要进行转置操作，对张量的列进行池化操作
x = x.transpose(1,2)
print(x.shape, "交换后")
#经过pooling层
y = layer(x)
print(y)
print(y.shape)
#squeeze方法去掉值为1的维度
y = y.squeeze()
print(y)
print(y.shape)
