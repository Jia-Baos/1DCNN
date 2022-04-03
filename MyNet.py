import torch
import torch.nn as nn


# 此文件负责网络的实现
class JiaNet(nn.Module):
    def __init__(self):
        super(JiaNet, self).__init__()

        self.c1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, padding=2)
        self.s1 = nn.Sigmoid()
        self.c2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, padding=2)
        self.s2 = nn.Sigmoid()
        self.c3 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=3, padding=2)
        self.s3 = nn.Sigmoid()
        self.l1 = nn.Linear(40 * 206, 200)
        self.l2 = nn.Linear(200, 50)
        self.l3 = nn.Linear(50, 2)
        self.s4 = nn.Softmax(dim=0)

    def forward(self, input):
        input = self.c1(input)
        input = self.s1(input)
        input = self.c2(input)
        input = self.s2(input)
        input = self.c3(input)
        input = self.s3(input)
        input = torch.flatten(input)
        input = self.l1(input)
        input = self.l2(input)
        input = self.l3(input)
        input = self.s4(input)

        return input


if __name__ == '__main__':
    src = torch.rand((1, 200))
    print(src.size())
    model = JiaNet()
    output = model(src)
    print(output.size())
    print(output)
