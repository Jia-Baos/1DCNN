import torch
from torch import nn
# 此文件负责一些必须函数的实现
# 例如：数据的预处理、loss的设计


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gama=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gama = gama

    def forward(self, preds, labels):
        tensor_temp1 = torch.zeros(preds.shape[0], dtype=torch.float)
        tensor_temp2 = torch.zeros(preds.shape[0], dtype=torch.float)

        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                if labels[i][j] == 1:
                    tensor_temp2[i] = -preds[i][j]
                tensor_temp1[i] += torch.exp(preds[i][j])
            tensor_temp1[i] = self.alpha * torch.pow((1 - (torch.exp(-tensor_temp2[i]) / tensor_temp1[i])), self.gama) * (
                                      tensor_temp2[i] + torch.log(tensor_temp1[i]))
        loss = 0.0
        for k in range(tensor_temp1.shape[0]):
            loss = (loss * k + tensor_temp1[k]) / (k + 1)

        return loss


if __name__ == '__main__':
    loss_func = FocalLoss()
    tensor1 = torch.tensor([[0.3669, 0.6331],
                            [0.2455, 0.7545],
                            [0.3984, 0.6016],
                            [0.2782, 0.7218]], dtype=torch.float)

    tensor2 = torch.tensor([[1., 0.],
                            [0., 1.],
                            [1., 0.],
                            [1., 0.]], dtype=torch.float)

    loss = loss_func.forward(tensor1, tensor2)
    print(loss)

    new_loss_func = nn.CrossEntropyLoss()
    new_loss = new_loss_func(tensor1, tensor2)
    print(new_loss)
