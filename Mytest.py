import torch
from MyNet import ResNet18
from MyDataSet import MyDataSet
from torch.utils.data import DataLoader

# 映射表
result_cls = ["Good", "Bad"]
# 此文件负责网络测试

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 权重存放路径
weight_path = "D:\\PythonProject\\1DCNN\\checkpoints\\best_model.pt"
# 调用模型
model = ResNet18().to(device)
model.load_state_dict(torch.load(weight_path))

if __name__ == '__main__':
    # 数据存放路径
    data_dir = 'D:\\PythonProject\\1DCNN\\data'
    # 加载测试数据集
    dataset = MyDataSet(data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    # 预测过程
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            print(output)
            cls_index = torch.argmax(output)
            print(result_cls[cls_index])
