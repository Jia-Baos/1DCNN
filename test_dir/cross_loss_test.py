import torch
import math
import torch.nn as nn

tensor1 = torch.tensor([[0.3669, 0.6331],
                        [0.2455, 0.7545],
                        [0.3984, 0.6016],
                        [0.2782, 0.7218]], dtype=torch.float)

tensor2 = torch.tensor([[1., 0.],
                        [0., 1.],
                        [1., 0.],
                        [1., 0.]], dtype=torch.float)

tensor_temp1 = torch.zeros(tensor1.shape[0], dtype=torch.float)
tensor_temp2 = torch.zeros(tensor1.shape[0], dtype=torch.float)

# CrossEntropyLoss
for i in range(tensor1.shape[0]):
    for j in range(tensor1.shape[1]):
        if tensor2[i][j] == 1:
            tensor_temp2[i] = -tensor1[i][j]
        tensor_temp1[i] += math.exp(tensor1[i][j])
    tensor_temp1[i] = tensor_temp2[i] + math.log(tensor_temp1[i])

my_loss = 0.0
for k in range(tensor_temp1.shape[0]):
    my_loss = (my_loss * k + tensor_temp1[k]) / (k + 1)

print("my_loss: ", my_loss)
loss_func = nn.CrossEntropyLoss()

output1 = loss_func(tensor1, tensor2)
print("output1: ", output1)


tensor_temp1 = torch.zeros(tensor1.shape[0], dtype=torch.float)
tensor_temp2 = torch.zeros(tensor1.shape[0], dtype=torch.float)
# focal loss
alpha = 0.25
gama = 2
for i in range(tensor1.shape[0]):
    for j in range(tensor1.shape[1]):
        if tensor2[i][j] == 1:
            tensor_temp2[i] = -tensor1[i][j]
        tensor_temp1[i] += math.exp(tensor1[i][j])
    tensor_temp1[i] = alpha * math.pow((1 - (math.exp(-tensor_temp2[i]) / tensor_temp1[i])), gama) * (
            tensor_temp2[i] + math.log(tensor_temp1[i]))

my_loss = 0.0
for k in range(tensor_temp1.shape[0]):
    my_loss = (my_loss * k + tensor_temp1[k]) / (k + 1)

print("my_focal_loss: ", my_loss)

# focal_loss used in pytorch
# 推导了一下公式，发现与其改造pytorch里面的函数，还不自己去写一个
