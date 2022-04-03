import torch
import numpy as np
import torch.nn as nn


# 此文件负责网络的实现
# 单源的一维数据无法实现batch处理
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
        # print("size of s1: {}".format(input.size()))

        input = self.c2(input)
        input = self.s2(input)
        # print("size of s2: {}".format(input.size()))

        input = self.c3(input)
        input = self.s3(input)
        # print("size of s3: {}".format(input.size()))

        input = torch.flatten(input)
        # print("size of s3 has flatten: {}".format(input.size()))

        input = self.l1(input)
        # print("size of l1: {}".format(input.size()))

        input = self.l2(input)
        # print("size of l2: {}".format(input.size()))

        input = self.l3(input)
        # print("size of l3: {}".format(input.size()))

        input = self.s4(input)
        # print("size of l3 has softmax: {}".format(input.size()))

        return input


if __name__ == '__main__':
    # src = torch.rand((1, 200))
    data_dir = "D:\\PythonProject\\1DCNN\\data\\Images\\000000.txt"
    src = np.loadtxt(data_dir)
    src = torch.tensor(src, dtype=torch.float)
    print("size of original src: {}".format(src.size()))
    src = src.unsqueeze(0)
    print("size of src has added dim: {}".format(src.size()))
    model = JiaNet()
    output = model(src)
    print("size of output: {}".format(output.size()))
    print("output of Net: {}".format(output))
