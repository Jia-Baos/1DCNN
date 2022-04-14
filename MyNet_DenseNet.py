import math
import torch
import torch.nn as nn
from MyDataSet import MyDataSet
from torch.utils.data import DataLoader


class Bottleneck(nn.Module):
    """
    Block内的一个小单元
    channels -> 4 * growth_rate -> growth_rate
    """

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_planes)
        self.conv1 = nn.Conv1d(in_channels=in_planes, out_channels=4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=4 * growth_rate)
        self.conv2 = nn.Conv1d(in_channels=4 * growth_rate, out_channels=growth_rate, kernel_size=3, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.bn1(x)
        output = self.relu(output)
        output = self.conv1(output)

        output = self.bn2(output)
        output = self.relu(output)
        output = self.conv2(output)

        # 在channels上进行叠加
        output = torch.cat([output, x], dim=1)

        return output


class Transition(nn.Module):
    """
    The transition is responsible for connection of two blocks
    """

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=in_planes)
        self.conv = nn.Conv1d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        output = self.bn(x)
        output = self.relu(output)
        output = self.conv(output)
        output = self.avg_pool(output)

        return output


class DenseNet(nn.Module):
    """
    block: which is equivalent to the Bottleneck
    num_block: which is responsible for the number of Bottleneck in Denseblock
    growth_rate: which is responsible for the increasement of channels
        Bottleneck: in_planes -> in_planes + growth_rate
        Denseblock: num_planes -> (num_planes + Denseblock's chennels * growth_rate) * reduction
    reduction: the ratio of downsample
    num_classes: the number of classes in result
    pool_size: the pool_size of pre_conv
    """

    def __init__(self, block, num_block, growth_rate=12, reduction=0.5, num_classes=10, pool_size=7):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.pool_size = pool_size

        # 进入DenseBlock前的预处理
        num_planes = 2 * growth_rate
        # channels = 1, the net is make for 1D signal
        self.conv1 = nn.Conv1d(1, num_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # channels = 24
        # the num_planes is equivalent to in_planes of Bottleneck
        self.dense1 = self._make_dense_layers(block, in_planes=num_planes, num_block=num_block[0])
        num_planes += num_block[0] * growth_rate
        # channels = 24 + 6 * 12 = 96
        # the reduction is equivalent the stride of avg_pool2d in Transition block
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(in_planes=num_planes, out_planes=out_planes)
        num_planes = out_planes

        # channels = 96 / 2 = 48
        self.dense2 = self._make_dense_layers(block, in_planes=num_planes, num_block=num_block[1])
        num_planes += num_block[1] * growth_rate
        # channels = 48 + 12 * 12 = 192
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(in_planes=num_planes, out_planes=out_planes)
        num_planes = out_planes

        # channels = 192 / 2 = 96
        self.dense3 = self._make_dense_layers(block, in_planes=num_planes, num_block=num_block[2])
        num_planes += num_block[2] * growth_rate
        # channels = 96 + 24 * 12 = 384
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(in_planes=num_planes, out_planes=out_planes)
        num_planes = out_planes

        # channels = 384 / 2 = 192
        self.dense4 = self._make_dense_layers(block, in_planes=num_planes, num_block=num_block[3])
        num_planes += num_block[3] * growth_rate
        # channels = 192 + 16 * 12 = 384
        self.bn = nn.BatchNorm1d(num_features=num_planes)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear = nn.Linear(in_features=num_planes, out_features=num_classes)

        for item in self.modules():
            if isinstance(item, nn.Conv1d):
                # n = item.kernel_size[0] * item.kernel_size[1] * item.out_channels
                n = item.kernel_size[0] * item.out_channels
                item.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(item, nn.BatchNorm1d):
                item.weight.data.fill_(1)
                item.bias.data.zero_()
            elif isinstance(item, nn.Linear):
                item.bias.data.zero_()

    def _make_dense_layers(self, block, in_planes, num_block):
        """
        in fact, the block in this func is the Bottleneck
        """
        layers = []
        for i in range(num_block):
            # 这里调用了Bottleneck类
            # block(in_planes, self.growth_rate) == Bottleneck(in_planes, self.growth_rate)
            layers.append(block(in_planes, self.growth_rate))
            # the result'channels of Bottleneck's forward is in_planes +grow_rate
            # so we need to adjust the in_planes of next Bottleneck
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input", x.size())
        x = self.conv1(x)
        # print("pre_conv: 7x7 S2", x.size())
        x = self.pool1(x)
        # print("pre_pool: 3x3 S2", x.size())
        x = self.trans1(self.dense1(x))
        # print("first_Denseblock: ", x.size())
        x = self.trans2(self.dense2(x))
        # print("second_Denseblock: ", x.size())
        x = self.trans3(self.dense3(x))
        # print("third_Denseblock: ", x.size())
        x = self.dense4(x)
        # print("fourth_Denseblock without transition:", x.size())
        x = self.avg_pool(self.relu(self.bn(x)))
        # print("global average pool: 7x7 ", x.size())
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size()[0], -1)
        # print("flatten the result of GAP: ", x.size())
        x = self.linear(x)
        x = nn.Softmax(dim=1)(x)
        # print("Softmax the output of linear layer: ", x.size())

        return x


def DenseNet121(pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12,
                    num_classes=num_classes, pool_size=pool_size)


def DenseNet161(pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=12,
                    num_classes=num_classes, pool_size=pool_size)


def DenseNet169(pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=12,
                    num_classes=num_classes, pool_size=pool_size)


def DenseNet201(pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=12,
                    num_classes=num_classes, pool_size=pool_size)


if __name__ == "__main__":
    model = DenseNet121(pool_size=7, num_classes=10)
    data_dir = "D:\\PythonProject\\1DCNN\\data"
    dataset = MyDataSet(data_dir)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0, drop_last=False)
    for batch, (key, value) in enumerate(dataloader):
        print(batch)
        print(key.size())
        print(value.size())
        output = model(key)
        print("size of output: {}".format(output.size()))
