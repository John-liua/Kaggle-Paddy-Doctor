# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: dataloader.py
# Time: 6/27/19 1:17 PM
# Description: 
# -------------------------------------------------------------------------------


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import os
import matplotlib.pyplot as plt
from dataload.preprocess import *
from dataload.constant import *
from dataload.mixup import *


########################################################################
# 正常分类训练的dataset, 返回图像和类别
#
class dataset(Dataset):
    def __init__(self, root, label, flag=1, signal=' ', transform=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._signal = signal
        self._list_images(self._root, self._label, self._signal)

    def _list_images(self, root, label, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0

        for line in label:
            cls = line.rstrip('\n').split(signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), float(cls[0])))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        label = self.items[index][1]
        if self._transform is not None:
            return self._transform(img), label
        return img, label


########################################################################
# 没有类别标签的dataset, 返回图像, 通常用于预测
#
class dataset_unlabeled(Dataset):
    def __init__(self, root, label, flag=1, transform=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._list_images(self._root, self._label)

    def _list_images(self, root, image_names):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in image_names:
            image_name = line.rstrip('\n')

            if os.path.isfile(os.path.join(root, image_name)):
                self.items.append((os.path.join(root, image_name), image_name))
            else:
                print(os.path.join(root, image_name))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        image_name = self.items[index][1]
        if self._transform is not None:
            return self._transform(img), image_name
        return img, image_name


########################################################################
# 读取label文件, 返回文件中所有的内容
#
def get_label(label_path):
    f = open(label_path)
    lines = f.readlines()
    return lines


########################################################################
# 加载正常分类训练的data, 预处理, 返回训练集和测试集
#
def load_data(root, train_paths, test_paths, signal=';', resize_size=(512, 512), input_size=(448, 448), batch_size=32,
              num_workers=0):
    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp
    test_list = []
    for i in test_paths:
        tmp = get_label(i)
        test_list = test_list + tmp

    train_transformer = transforms.Compose([
        # RandomResize(resize_size),
        RandomResizePadding(resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    test_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    train_dataset = dataset(root, train_list, flag=1, signal=signal, transform=train_transformer)
    test_dataset = dataset(root, test_list, flag=1, signal=signal, transform=test_transformer)

    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_iter = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_iter, test_iter


########################################################################
# 加载没有类别标签的data, 预处理, 返回测试集
#
def load_unlabeled_data(root, test_paths, resize_size=(512, 512), input_size=(448, 448), batch_size=32,
                        num_workers=0):
    test_list = []
    for i in test_paths:
        tmp = get_label(i)
        test_list = test_list + tmp

    test_transformer = transforms.Compose([
        transforms.Resize(resize_size, Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    test_dataset = dataset_unlabeled(root, test_list, flag=1, transform=test_transformer)

    test_iter = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_iter


########################################################################
# 测试函数是否正确
#
if __name__ == '__main__':
    train_iter, test_iter = load_data(
        root='./database',
        train_paths=['../datalist/train.txt'],
        test_paths=['../datalist/test.txt'],
        signal=' ',
        resize_size=(512, 512),
        input_size=(448, 448),
        batch_size=2,
        num_workers=2
    )

    for data in train_iter:
        inputs, labels = data
        image = np.transpose(inputs[0], (2, 1, 0))
        plt.imshow(image)
        plt.show()
        break
    pass
