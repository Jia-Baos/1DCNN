import os
import random
import xml.etree.ElementTree as ET

classes = ['mask', 'no-mask']  # 根据自己的类别来
data_dir = r'D:\PythonProject\1DCNN\data'


# 统一更改图片名称，也可以用其来修改xml文件的名称
def change_name(images_dir):
    count = 0
    filelist = os.listdir(images_dir)
    for file in filelist:
        print(file)
        oldname = os.path.join(images_dir, file)
        # 如果是文件夹则跳过
        if os.path.isdir(oldname):
            continue
        filename = os.path.splitext(file)[0]
        filetype = os.path.splitext(file)[1]
        newname = os.path.join(images_dir, str(count).zfill(6) + filetype)
        os.rename(oldname, newname)
        count += 1
    print("change_name has done!!!")


# 将标注由.xml转化为.txt
def change_label(labels_xml_dir, labels_txt_dir):
    xml_list = os.listdir(labels_xml_dir)
    for file in xml_list:
        print(file)
        # 获取xml文件的name
        xml_id = os.path.splitext(file)[0]
        # 获取xml文件的路径
        xml_path = os.path.join(labels_xml_dir, file)
        # 对xml_path文件构建tree
        xml_tree = ET.parse(xml_path)
        # 获取根节点
        xml_root = xml_tree.getroot()
        # 获取图像的尺寸
        size = xml_root.find('size')

        f = open(os.path.join(labels_txt_dir, xml_id + '.txt'), 'w', encoding='utf-8')
        if size != None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            # 获取Ground Truth的cls,x1,x2,y1,y2
            for obj in xml_root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                # 获取object类别对应的索引
                cls_id = classes.index(cls)
                obj_box = obj.find('bndbox')
                # xml文件坐标系与opencv坐标系相反
                points = (
                    float(obj_box.find('xmin').text),
                    float(obj_box.find('ymin').text),
                    float(obj_box.find('xmax').text),
                    float(obj_box.find('ymax').text)
                )
                print(xml_id, cls, points)
                f.write(str(cls_id) + " " + " ".join([str(item) for item in points]) + '\n')
        f.close()
    print("change_label has done!!!")


# 划分训练集、验证集、测试集
def divide_sets(labels_txt_dir, imagesets_dir):
    ftrain = open(os.path.join(imagesets_dir, 'train.txt'), 'w')
    fval = open(os.path.join(imagesets_dir, 'val.txt'), 'w')
    ftest = open(os.path.join(imagesets_dir, 'test.txt'), 'w')

    trainval_percent = 0.9
    train_percent = 0.9

    total_xml = os.listdir(labels_txt_dir)
    xml_num = len(total_xml)
    list = range(xml_num)
    trainval_num = int(xml_num * trainval_percent)
    train_num = int(trainval_num * train_percent)

    trainval = random.sample(list, trainval_num)
    train = random.sample(trainval, train_num)

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrain.close()
    fval.close()
    ftest.close()
    print("divide_sets has done!!!")


if __name__ == '__main__':
    # 所有图片所在文件夹
    images_dir = os.path.join(data_dir, "Images")

    # 新建文件夹存放.txt标注文件
    labels_txt_dir = os.path.join(data_dir, "Labels")
    if not os.path.exists(labels_txt_dir):
        os.makedirs(labels_txt_dir)

    # 新建文件夹存放train.txt、val.txt、test.txt文件
    imagesets_dir = os.path.join(data_dir, "ImageSets")
    if not os.path.exists(imagesets_dir):
        os.makedirs(imagesets_dir)

    # change_name(labels_txt_dir)
    divide_sets(labels_txt_dir, imagesets_dir)
