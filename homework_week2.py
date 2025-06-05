"""
基于pytorch框架编写训练模型
实现一个自行构造的找规律（机器学习）任务
五维判断：x是一个5维向量，向量中哪个标量最大就输出哪一维下标
2分类：输出0-1一个数，大于0.5正类，否则负类
多分类：输出概率分布，几分类就输出几维的向量，哪一类的概率最大结果为哪一类
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MultiClassficationModel(nn.Module):
    def __init__(self,input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear=nn.Linear(input_size,5)      #线性层，输出为5维的向量
        self.loss=nn.functional.cross_entropy #loss函数采用交叉熵损失函数

    def forward(self,x,y=None):
        y_pred=self.linear(x)  #(batch_size,input_size)->(batch_size,1)
        if y is not None:
            loss=self.loss(y_pred,y) #预测值真实值计算损失
            return loss
        else:
            return y_pred

#生成样本
# 随机生成5维向量，根据每个向量中最大的标量同一标构建Y        
def build_sample():
    x=np.random.random(5)
    #获得最大索引
    max_index=np.argmax(x)
    return x,max_index

#随机生成一批样本
#正负样本均匀生成
def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)

#训练模型
def evaluate(model):
    model.eval()
    total_sample_num = 100
    x, y = build_dataset(total_sample_num)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p).item() == y_t.item():
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

#训练模型
def main():
    # 配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 5000    #每轮训练总共训练的样本总数
    input_size=5          #输入向量维度
    learning_rate = 0.001 #学习率

    # 创建数据集，正常任务是读取数据集
    train_x, train_y = build_dataset(train_sample)

    # 建立模型
    model =MultiClassficationModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(int(train_sample // batch_size)):
            
            x=train_x[batch_index*batch_size:(batch_index+1)*batch_size ] #取出一批训练数据
            y=train_y[batch_index*batch_size:(batch_index+1)*batch_size] #取出一批训练标签
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            optim.zero_grad()  # 清空梯度
            loss = model(x,y)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    plt.plot([i[0] for i in log], label="Accuracy")
    plt.plot([i[1] for i in log], label="Loss")
    plt.legend()
    plt.show()  
    return

if __name__ == '__main__':
    main()
  




