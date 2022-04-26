import torch
import torch.nn as nn

preds = torch.tensor([[0.3669, 0.6331],
                      [0.2455, 0.7545],
                      [0.3984, 0.6016],
                      [0.2782, 0.7218]], dtype=torch.float, requires_grad=True)

labels = torch.tensor([[1., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 0.]], dtype=torch.float, requires_grad=True)

preds_exp = torch.exp(preds)
print(preds_exp)
preds_exp_sum = torch.sum(preds_exp, dim=1)
print(preds_exp_sum)
real_class_index = torch.argmax(labels, dim=1)
print(real_class_index)
real_class_index = torch.unsqueeze(real_class_index, dim=1)
print(real_class_index)
real_class_probability = preds_exp.gather(dim=1, index=real_class_index)
print(real_class_probability)
preds_exp_sum = torch.unsqueeze(preds_exp_sum, dim=1)
print(preds_exp_sum)
temp = real_class_probability / preds_exp_sum
loss = -1 * 0.25 * torch.pow((1 - temp), 2) * torch.log(temp)

loss_num = torch.sum(loss)
print(loss_num)
