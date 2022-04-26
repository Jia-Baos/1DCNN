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
        loss_item = 0.0
        preds_exp = torch.exp(preds)
        preds_exp_sum = torch.sum(preds_exp, dim=1)
        real_class_index = torch.argmax(labels, dim=1)
        real_class_index = torch.unsqueeze(real_class_index, dim=1)
        real_class_probability = preds_exp.gather(dim=1, index=real_class_index)
        preds_exp_sum = torch.unsqueeze(preds_exp_sum, dim=1)
        temp = real_class_probability / preds_exp_sum
        loss = -1 * self.alpha * torch.pow((1 - temp), self.gama) * torch.log(temp)

        loss_item = torch.sum(loss)
        return loss_item


if __name__ == '__main__':
    loss_func = FocalLoss()
    tensor1 = torch.tensor([[0.3669, 0.6331],
                            [0.2455, 0.7545],
                            [0.3984, 0.6016],
                            [0.2782, 0.7218]], dtype=torch.float, requires_grad=True)

    tensor2 = torch.tensor([[1., 0.],
                            [0., 1.],
                            [1., 0.],
                            [1., 0.]], dtype=torch.float, requires_grad=True)

    loss = loss_func.forward(tensor1, tensor2)
    print("focal loss: ", loss)
    loss.backward()
    print("focal loss's grad: ", tensor1.grad)
    print("focal loss's grad: ", tensor2.grad)

    new_loss_func = nn.CrossEntropyLoss()
    new_loss = new_loss_func(tensor1, tensor2)
    print(new_loss)
