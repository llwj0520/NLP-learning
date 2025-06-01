import torch
import torch.nn as nn
import random
import string
import numpy as np
import json


# 定义模型
class TorchModel(nn.Module):
    def __init__(self,vector_dim,sentence_length,vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim, padding_idx=0)  # 字符嵌入层
        self.pool=nn.AvgPool1d(sentence_length) # 池化层
        # RNN层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  #输入和隐藏层的维度均为 vector_dim

        # +1的原因是可能出现a不存在的情况，那时的真实label在构造数据时设为了sentence_length
        self.classify = nn.Linear(vector_dim, sentence_length+1)  #将模型的输出特征向量映射到一个长度为 sentence_length + 1 的向量，即分类任务的输出结果。
        self.loss = nn.functional.cross_entropy  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        #使用pooling的情况
        # x = x.transpose(1, 2)           
        # x = self.pool(x)                
        # x = x.squeeze()   
        #使用rnn的情况 
        run_out, hidden = self.rnn(x)  # RNN处理输入
        x= run_out[:, -1, :]  #或者写hidden.squeeze()也是可以的，因为rnn的hidden就是最后一个位置的输出

        #接线性层做分类
        y_pred = self.classify(x)  # 全连接层
        if y is not None:
            return self.loss(y_pred, y)   # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefjhijkg"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   # 每个字对应一个序号
    vocab['unk'] = len(vocab)  #11 "unk" 用于未识别的字符
    return vocab


# 随机生成一个样本，查找a的位置
def build_sample(vocab, sentence_length):
    #注意这里用sample，是不放回的采样，每个字母不会重复出现，但是要求字符串长度要小于词表长度
    x = random.sample(list(vocab.keys()), sentence_length)
    # 查找a的位置
    if 'a' in x:
        y=x.index('a')
    else:
        y=sentence_length
    # 将字符转化为序号，从而实现embedding
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 建立数据集
def build_dataset(sample_size, vocab, sentence_length): #sample_size样本数量
    dataset_x = []
    dataset_y = []
    for i in range(sample_size):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, vector_dim, sentence_length):#vector_dim词向量维度（embedding 输出维度）
    model = TorchModel(vector_dim, sentence_length, vocab)
    return model


# 评估模型准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 主函数
def main():
    # 配置参数
    epoch_num = 20        #训练轮数
    batch_size = 40       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    vector_dim = 30         #每个字的维度
    sentence_length = 10   #样本文本统一截断或填充（padding）后的长度
    learning_rate = 0.001 #学习率

    # 建立字表
    vocab = build_vocab()

    # 建立模型
    model = build_model(vocab,vector_dim,sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构建一个批次的数据
            optim.zero_grad()  # 清空梯度
            loss = model(x, y)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 预测函数
def predict(model_path, vocab_path, input_strings):
     vector_dim = 30  # 每个字的维度
     sentence_length = 10  # 样本文本长度
     vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
     model = build_model(vocab, vector_dim, sentence_length)     #建立模型
     model.load_state_dict(torch.load(model_path))             #加载训练好的权重
     x = []
     for input_string in input_strings:
         x.append([vocab[char] for char in input_string])  #将输入序列化
     model.eval()   #测试模式
     with torch.no_grad():  #不计算梯度
         result = model.forward(torch.LongTensor(x))  #模型预测
     for i, input_string in enumerate(input_strings):
         print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict("model.pth", "vocab.json", test_strings)