import torch
import numpy as np
import torch.nn as nn

data_dir = "/data/Images/000000.txt"
src = np.loadtxt(data_dir)
print(src)
print(src.shape)
print(src[0])
print(type(src[0]))

t1 = torch.rand((1, 2))
t2 = torch.rand(2)
t1 = t1.unsqueeze(0)
print(t1)
print(t2)

from visdom import Visdom
import time
# 将窗口类实例化
viz = Visdom()
# 创建窗口并初始化
viz.line([[0.,0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))
for global_steps in range(100):
    # 随机获取loss和acc
    loss = 0.1 * np.random.randn() + 1
    acc = 0.1 * np.random.randn() + 0.5
    # 更新窗口图像
    viz.line([[loss, acc]], [global_steps], win='train', update='append')
    # 延时0.5s
    time.sleep(0.5)
