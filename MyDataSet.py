import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 此文件负责封装DataSet，便于数据按bath的方式读取
class MyDataSet(Dataset):
    """
        dataset_dir
        ----Images
        --------000000.txt
        ------------标量1
        ------------标量2
                    ~
        ------------标量200
        --------000001.txt
                    ~
        ----ImageSets
        --------val.txt
        --------text.txt
        --------train.txt
                    ~
        ----Labels
        --------000000.txt
        ------------1 0
                    or
        ------------0 1
        --------000001.txt
                    ~
    """

    def __init__(self, dataset_dir, mode="train", trans=None):
        self.data_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "Images_min_max_normalize")
        self.label_dir = os.path.join(dataset_dir, "Labels")
        self.imagesets_dir = os.path.join(dataset_dir, "ImageSets")
        self.mode = mode

        # img_list存的只是图片的名字
        self.img_list = []
        if mode == "train":
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        elif mode == "val":
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        else:
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        self.trans = trans

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.img_list[item] + '.txt')
        img = np.loadtxt(img_path)
        label_path = os.path.join(self.label_dir, self.img_list[item] + '.txt')
        label = np.loadtxt(label_path)
        image = torch.tensor(img, dtype=torch.float)
        image = image.unsqueeze(0)
        # image = image.unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float)
        return image, label


if __name__ == '__main__':
    data_dir = r'D:\PythonProject\1DCNN\data'
    dataset = MyDataSet(data_dir)
    # DataLoader要求输入图片大小一致
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
    for batch, (x, y) in enumerate(dataloader):
        print(batch)
        print(x.size())
        print(y.size())
