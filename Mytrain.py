# 此文件负责网络训练
import torch
from torch import nn
from MyNet_ResNet import ResNet18
from MyNet_SEResNet import SEResNet18
from MyNet_DenseNet import Bottleneck, DenseNet
from MyDataSet import MyDataSet
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from Myutils import FocalLoss

import os
import math
import time
import datetime
from visdom import Visdom

# 映射表
result_cls = ["Good", "Bad"]

# 数据存放路径
data_dir = 'D:\\PythonProject\\1DCNN\\data'
# 权重存放路径
checkpoints_dir = "D:\\PythonProject\\1DCNN\\checkpoints"

# 加载训练数据集
dataset = MyDataSet(data_dir, mode='train')
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 加载验证数据集
dataset = MyDataSet(data_dir, mode='val')
val_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用MyNet模型，将模型数据转到GPU
# model = ResNet18().to(device)
model = SEResNet18().to(device)
# model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=2, pool_size=7).to(device)
model.load_state_dict(torch.load("D:\\PythonProject\\1DCNN\\checkpoints\\best_model_SEResNet_min_max.pt"))

# 定义损失函数
loss_func = nn.CrossEntropyLoss()
new_loss_func = FocalLoss()

# 定义一个优化器
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 学习率每隔十轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_func, optimizer):
    model.train()
    train_avg_loss = 0.0
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        x, y = x.to(device), y.to(device)
        output = model(x)
        # output = output.unsqueeze(0)
        print(output)
        print(y)
        cur_loss = loss_func(output, y)
        # cur_loss = new_loss_func.forward(output, y)
        # cur_loss = ()
        train_avg_loss = (train_avg_loss * batch + cur_loss.item()) / (batch + 1)
        # pred = torch.argmax(output, dim=1)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

    log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
    log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, train_avg_loss))
    log_file.flush()
    log_file.close()

    return train_avg_loss


def val(dataloader, model, loss_func):
    model.eval()
    val_avg_loss = 0.0
    num_total, num_right = 0.0, 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)
            # output = output.unsqueeze(0)
            cur_loss = loss_func(output, y)
            # cur_loss = new_loss_func.forward(output, y)
            val_avg_loss = (val_avg_loss * batch + cur_loss.item()) / (batch + 1)
            # pred = torch.argmax(output, dim=1)
            real_cls_index = torch.argmax(y)
            cls_index = torch.argmax(output)
            print("***************************************")
            print(result_cls[real_cls_index])
            print(result_cls[cls_index])
            if real_cls_index == cls_index:
                num_right += 1
            num_total += 1

        log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
        log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, val_avg_loss))
        log_file.flush()
        log_file.close()

    return val_avg_loss, num_right/num_total


# 开始训练
if __name__ == '__main__':

    epochs = 100
    best_loss = 2.0

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # 将窗口类实例化
    # 在终端中按下Ctrl + C可以终止前端服务器
    viz = Visdom()
    # 创建窗口并初始化
    viz.line([[0., 0., 0.]], [0], win='train',
             opts=dict(title='train_loss&val_loss&accur', legend=['train_loss', 'val_loss', 'accur']))

    for epoch in range(epochs):
        print("\nepoch: %d" % (epoch + 1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            # strftime()，格式化输出时间
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
            log_file.write(localtime)
            log_file.write("\n======================training epoch %d======================\n" % (epoch + 1))

        t1 = time.time()
        train_loss = train(train_dataloader, model, loss_func, optimizer)
        t2 = time.time()

        print("Training consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Training consumes %.2f second\n" % (t2 - t1))

        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("\n======================validate epoch %d======================\n" % (epoch + 1))
        t1 = time.time()
        val_loss, accur = val(val_dataloader, model, loss_func)
        t2 = time.time()

        # 更新学习率
        lr_scheduler.step()
        print(epoch, lr_scheduler.get_last_lr()[0])

        # 更新窗口图像
        viz.line([[train_loss, val_loss, accur]], [epoch], win='train', update='append')
        # viz.line([train_loss, val_loss], [epoch], win='loss', update='append')
        time.sleep(0.5)

        print("Validation consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Validation consumes %.2f second\n\n" % (t2 - t1))

        # 保存最好的模型权重
        if val_loss < best_loss:
            best_loss = val_loss
            print("save best model")
            torch.save(model.state_dict(), 'checkpoints/best_model_2022_04_23.pt')
    print("The train has done!!!")
