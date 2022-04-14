import os
import numpy as np

data_path = "D:\\PythonProject\\1DCNN\\data\\Images"
data_save_path = "D:\\PythonProject\\1DCNN\\data\\Imagescopy"

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)


def min_max_normalize():
    for item in os.listdir(data_path):
        img_path = os.path.join(data_path, item)
        img_save_path = os.path.join(data_save_path, item)
        content = np.loadtxt(img_path)
        min_content = np.min(content)
        max_content = np.max(content)

        with open(img_save_path, 'w') as f:
            for value in content:
                new_value = (value - min_content) / (max_content - min_content)
                f.write(str(new_value))
                f.write('\n')
        f.close()


def z_zero_normalize():
    for item in os.listdir(data_path):
        img_path = os.path.join(data_path, item)
        img_save_path = os.path.join(data_save_path, item)
        content = np.loadtxt(img_path)
        mean_content = np.mean(content)
        std_content = np.std(content)

        with open(img_save_path, 'w') as f:
            for value in content:
                new_value = (value - mean_content) / std_content
                f.write(str(new_value))
                f.write('\n')
        f.close()


if __name__ == '__main__':
    z_zero_normalize()
