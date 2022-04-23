import os
import numpy as np

data_dir = "D:\\PythonProject\\1DCNN\\data"

file_list_path = os.path.join(data_dir, "Images_200")
file_list = os.listdir(file_list_path)

for file in file_list:
    file_path = os.path.join(file_list_path, file)
    print(file_path)
    with open(file_path, 'r') as f1:
        content = np.loadtxt(f1)
        print(np.size(content))
