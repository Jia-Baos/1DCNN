import os
import torch
import numpy as np
from MyNet import JiaNet

# 映射表
result_cls = ["Good", "Bad"]
# 此文件负责网络测试

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 权重存放路径
weight_path = "D:\\PythonProject\\1DCNN\\checkpoints\\best_model.pt"
# 调用模型
model = JiaNet().to(device)
model.load_state_dict(torch.load(weight_path))

if __name__ == '__main__':
    # 数据存放路径
    data_dir = 'D:\\PythonProject\\1DCNN\\datatest'

    # 预测过程
    img_list = os.listdir(data_dir)
    for img_name in img_list:
        print(img_name)
        img_path = os.path.join(data_dir, img_name)
        img = np.loadtxt(img_path)
        image = torch.tensor(img, dtype=torch.float)
        image = image.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            x = image.to(device)
            output = model(x)
            print(output)
            cls_index = torch.argmax(output)
            print(result_cls[cls_index])

