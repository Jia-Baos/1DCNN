# 此文件负责网络训练
import torch
from torch import nn
from MyNet import JiaNet
from MyDataSet import MyDataSet
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import datetime

data_dir = 'D:\\PythonProject\\1DCNN\\data'
checkpoints_dir = "D:\\PythonProject\\1DCNN\\checkpoints"

# 加载训练数据集
dataset = MyDataSet(data_dir, mode='train')
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

# 加载测试数据集
dataset = MyDataSet(data_dir, mode='val')
val_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用MyNet模型，将模型数据转到GPU
model = JiaNet().to(device)
# model.load_state_dict(torch.load("D:\\PythonProject\\LeNet\\save_model\\2020-03-06.pkl"))

# 定义损失函数
loss_func = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔十轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_func, optimizer):
    avg_loss = 0.0
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        x, y = x.to(device), y.to(device)
        output = model(x)
        output = output.unsqueeze(0)
        print(output)
        print(y)
        cur_loss = loss_func(output, y)
        avg_loss = (avg_loss * batch + cur_loss.item()) / (batch + 1)
        # pred = torch.argmax(output, dim=1)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

    log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
    log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, avg_loss))
    log_file.flush()
    log_file.close()


def val(dataloader, model, loss_func):
    model.eval()
    avg_loss = 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)
            output = output.unsqueeze(0)
            cur_loss = loss_func(output, y)
            avg_loss = (avg_loss * batch + cur_loss.item()) / (batch + 1)
            # pred = torch.argmax(output, dim=1)

        log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
        log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, avg_loss))
        log_file.flush()
        log_file.close()

    return avg_loss


# 开始训练
if __name__ == '__main__':

    epochs = 50
    best_loss = 2.0

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    for epoch in range(epochs):
        print("\nepoch: %d" % (epoch + 1))

        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            # strftime()，格式化输出时间
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
            log_file.write(localtime)
            log_file.write("\n======================training epoch %d======================\n" % (epoch + 1))

        t1 = time.time()
        train(train_dataloader, model, loss_func, optimizer)
        t2 = time.time()

        print("Training consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Training consumes %.2f second\n" % (t2 - t1))

        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("\n======================validate epoch %d======================\n" % (epoch + 1))
        t1 = time.time()
        val_loss = val(val_dataloader, model, loss_func)
        t2 = time.time()

        print("Validation consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Validation consumes %.2f second\n\n" % (t2 - t1))

        # 保存最好的模型权重
        if val_loss < best_loss:
            best_loss = val_loss
            print("save best model\n")
            torch.save(model.state_dict(), 'checkpoints/best_model.pkl')
    print("The train has done!!!")
