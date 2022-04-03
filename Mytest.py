import os
import numpy as np

data_dir = "D:\\PythonProject\\1DCNN\\data\\Images\\000000.txt"
src = np.loadtxt(data_dir)
print(src)
print(src.shape)
print(src[0])
print(type(src[0]))
