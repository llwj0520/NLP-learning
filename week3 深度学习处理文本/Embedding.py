#coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
'''

num_embeddings = 8  #通常对于nlp任务，此参数为字符集字符总数
embedding_dim = 5   #每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
print("随机初始化权重")
print(embedding_layer.weight)
print("################")
 
#构造字符表
vocab = {
    "[pad]" : 0,
    "e" : 1,
    "中国" : 2,
    "c" : 3,
    "d" : 4,
    "a" : 5,
    "f" : 6,
    "[unk]":7 #将不认识的字符都看作为[unk]，通常是一个特殊字符
}

#你好中国 -> 你好 中国 -> [3,2]

#字符串按照字符表转化为序列
def str_to_sequence(string, vocab):
    seq = [vocab.get(s, vocab["[unk]"]) for s in string][:5]
    if len(seq) < 5: # 最大长度应该设置为超过训练集99%的数据长度 #使不同长度的序列能够变成同样大小的张量进入模型训练
        seq += [vocab["[pad]"]] * (5 - len(seq))
    return seq

string1 = "abcd"
string2 = "ddcc"
string3 = "feda"

sequence1 = str_to_sequence(string1, vocab)
sequence2 = str_to_sequence(string2, vocab)
sequence3 = str_to_sequence(string3, vocab)

print(sequence1)
print(sequence2)
print(sequence3)

x = torch.LongTensor([sequence1, sequence2, sequence3])
embedding_out = embedding_layer(x)
print(embedding_out)


# 为了让不同长度的训练样本能够放在同一个batch中，需要将所有样本补齐或截断到相同长度
# padding 补齐
# [1,2,3,0,0]
# [1,2,3,4,0]
# [1,2,3,4,5]
# 截断
# [1,2,3,4,5,6,7] -> [1,2,3,4,5]
