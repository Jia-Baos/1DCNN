import torch
import numpy as np
import torch.nn as nn
from MyDataSet import MyDataSet
from torch.utils.data import DataLoader


# ResNet版本的一维卷积网络
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        # print("ResNetBasicBlock 1: ", output.size())

        output = self.conv2(output)
        output = self.bn2(output)
        # print("ResNetBasicBlock 2: ", output.size())

        output = self.relu(x + output)
        # print("ResNetBasicBlock 3: ", output.size())

        return output


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.extra = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        # print("ResNetDownBlock 1: ", output.size())

        output = self.conv2(output)
        output = self.bn2(output)
        # print("ResNetDownBlock 2: ", output.size())

        output = self.relu(extra_x + output)
        # print("ResNetDownBlock 3: ", output.size())

        return output


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64, 1),
                                    ResNetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(ResNetDownBlock(64, 128, [2, 1]),
                                    ResNetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, [2, 1]),
                                    ResNetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(ResNetDownBlock(256, 512, [2, 1]),
                                    ResNetBasicBlock(512, 512, 1))

        # 如此可以固定全连接层的输入
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        output = self.conv1(x)
        # print("first conv: ", output.size())
        output = self.bn1(output)
        output = self.maxpool(output)
        # print("first maxpool: ", output.size())

        output = self.layer1(output)
        # print("first Resnetblock: ", output.size())
        output = self.layer2(output)
        # print("second Resnetblock: ", output.size())
        output = self.layer3(output)
        # print("third Resnetblock: ", output.size())
        output = self.layer4(output)
        # print("fourth Resnetblock: ", output.size())

        output = self.avgpool(output)
        output = output.reshape(x.shape[0], -1)
        # print("ready enter fc: ", output.size())

        output = self.fc(output)
        # print("output of fc: ", output.size())
        return output

# 此文件负责网络的实现
# 单源的一维数据无法实现batch处理，该问题目前已经解决
class JiaNet(nn.Module):
    def __init__(self):
        super(JiaNet, self).__init__()

        # 通过池化层，层数增加来增大感受野，也可以寻找现有的一维网络
        # LSTM考虑数据是否存在时序性
        # 特征泛化，考虑旋转的抗性（1、数据增强；2、网络调整）
        # 任务：
        #       1、构造数据集，图形化界面
        #       2、
        self.c1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.s1 = nn.Sigmoid()
        self.c2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.s2 = nn.Sigmoid()
        self.c3 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=3, padding=1)
        self.s3 = nn.Sigmoid()
        self.l1 = nn.Linear(40 * 200, 200)
        self.l2 = nn.Linear(200, 50)
        self.l3 = nn.Linear(50, 2)
        self.s4 = nn.Softmax(dim=0)

    def forward(self, input):
        input = self.c1(input)
        input = self.s1(input)
        print("size of s1: {}".format(input.size()))

        input = self.c2(input)
        input = self.s2(input)
        print("size of s2: {}".format(input.size()))

        input = self.c3(input)
        input = self.s3(input)
        print("size of s3: {}".format(input.size()))

        input = torch.flatten(input, 1)
        print("size of s3 has flatten: {}".format(input.size()))

        input = self.l1(input)
        print("size of l1: {}".format(input.size()))

        input = self.l2(input)
        print("size of l2: {}".format(input.size()))

        input = self.l3(input)
        print("size of l3: {}".format(input.size()))

        input = self.s4(input)
        print("size of l3 has softmax: {}".format(input.size()))

        return input

if __name__ == '__main__':
    # src = torch.rand((1, 1, 200), dtype=torch.float)
    # data_dir = "D:\\PythonProject\\1DCNN\\data\\Images\\000000.txt"
    # src = np.loadtxt(data_dir)
    # src = torch.tensor(src, dtype=torch.float)
    # print("size of original src: {}".format(src.size()))
    # src = src.unsqueeze(0)
    # src = src.unsqueeze(0)
    # print("size of src has added dim: {}".format(src.size()))
    # model = JiaNet()
    # output = model(src)
    # print("size of output: {}".format(output.size()))
    # print("output of Net: {}".format(output))

    # model = JiaNet()
    model = ResNet18()
    data_dir = "D:\\PythonProject\\1DCNN\\data"
    dataset = MyDataSet(data_dir)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
    for batch, (key, value) in enumerate(dataloader):
        print(batch)
        print(key.size())
        print(value.size())
        output  = model(key)
        print("size of output: {}".format(output.size()))
        print("output of Net: {}".format(output))


