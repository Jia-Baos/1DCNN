import torch
from MyNet_ResNet import ResNet18
from MyNet_SEResNet import SEResNet18
from MyNet_DenseNet import Bottleneck, DenseNet
from MyDataSetcopy import MyDataSet
from torch.utils.data import DataLoader

# 映射表
result_cls = ["Good", "Bad"]
# 此文件负责网络测试

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 权重存放路径
weight_path = "D:\\PythonProject\\1DCNN\\checkpoints\\best_model_SEResNet_min_max.pt"
# 调用模型
# model = ResNet18().to(device)
model = SEResNet18().to(device)
# model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=2, pool_size=7).to(device)
model.load_state_dict(torch.load(weight_path))

if __name__ == '__main__':
    # 数据存放路径
    data_dir = 'D:\\PythonProject\\1DCNN\\data'
    # 加载测试数据集
    dataset = MyDataSet(data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    # 预测过程
    data_right, data_full = 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            # print(output)
            real_cls_index = torch.argmax(y)
            cls_index = torch.argmax(output)
            print("***************************************")
            print(result_cls[real_cls_index])
            print(result_cls[cls_index])
            if real_cls_index == cls_index:
                data_right += 1
            data_full += 1

    print("accur: ", data_right / data_full)
