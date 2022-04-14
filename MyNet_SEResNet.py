import torch
import torch.nn as nn
from MyDataSet import MyDataSet
from torch.utils.data import DataLoader


# 1DSENet
class SELayer(nn.Module):
    """
    The mechanism of SELayer is giving every channel a attention weight, so the output of SELayer is the input multiple attention weights
    the Squeeze and Excitation realizes the interaction of features from different channels
    the activation func of Sigmoid realizes the flexibility of weights, which means several channels can embrace the high weights at the same time
    """

    def __init__(self, in_channels=8, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.reshape(x.shape[0], x.shape[1])
        # print("avg_pool's size: ", y.size())
        # print("avg_pool: ", y)

        y = self.fc(y)
        y = y.reshape(x.shape[0], x.shape[1], 1)
        # print("Squeeze and Excitation's size: ", y.size())
        # print("Squeeze and Excitation: ", y)

        return x * y


# ResNet版本的一维卷积网络
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.selayer = SELayer(in_channels=out_channels, reduction=4)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        # print("ResNetBasicBlock 1: ", output.size())

        output = self.selayer(output)
        # print("SEBlock: ", output.size())

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
        self.selayer = SELayer(in_channels=out_channels, reduction=4)

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

        output = self.selayer(output)
        # print("SEBlock: ", output.size())

        output = self.conv2(output)
        output = self.bn2(output)
        # print("ResNetDownBlock 2: ", output.size())

        output = self.relu(extra_x + output)
        # print("ResNetDownBlock 3: ", output.size())

        return output


class SEResNet18(nn.Module):
    def __init__(self):
        super(SEResNet18, self).__init__()
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


if __name__ == '__main__':
    model = SEResNet18()
    data_dir = "D:\\PythonProject\\1DCNN\\data"
    dataset = MyDataSet(data_dir)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0, drop_last=False)
    for batch, (key, value) in enumerate(dataloader):
        print(batch)
        print(key.size())
        print(value.size())
        output = model(key)
        print("size of output: {}".format(output.size()))
        print("output of Net: {}".format(output))
