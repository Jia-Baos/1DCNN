import os
import torch
import numpy as np
from MyNet_ResNet import ResNet18
from MyNet_SEResNet import SEResNet18
from MyNet_DenseNet import Bottleneck, DenseNet

# 映射表
result_cls = ["Good", "Bad"]
# 此文件负责网络测试

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 权重存放路径
weight_path = "D:\\PythonProject\\1DCNN\\checkpoints\\best_model_SEResNet.pt"
# 调用模型
# model = ResNet18().to(device)
model = SEResNet18().to(device)
# model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=2, pool_size=7).to(device)
model.load_state_dict(torch.load(weight_path))

if __name__ == '__main__':
    # 数据存放路径
    data_dir = 'D:\\PythonProject\\1DCNN\\datatest'

    # 预测过程
    img_list = os.listdir(data_dir)
    data_right, data_full = 0, 0
    for img_name in img_list:
        print(img_name)
        img_path = os.path.join(data_dir, img_name)
        img = np.loadtxt(img_path)
        image = torch.tensor(img, dtype=torch.float)
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            x = image.to(device)
            output = model(x)
            cls_index = torch.argmax(output)
            print(result_cls[cls_index])
            if cls_index == 1:
                data_right += 1
            data_full += 1
    print("accur: ", data_right / data_full)
