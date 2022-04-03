import torch
import numpy as np

data_dir = "D:\\PythonProject\\1DCNN\\data\\Images\\000000.txt"
src = np.loadtxt(data_dir)
print(src)
print(src.shape)
print(src[0])
print(type(src[0]))

t1 = torch.rand((1,2))
t2 = torch.rand(2)
t1 = t1.squeeze(0)
print(t1)
print(t2)
