import torch
import torch.nn as nn
from torch import optim
import numpy as np
# from torch.utils.data import DataLoader, Dataset
import torchvision.models
from torchvision.transforms import transforms
import torchvision.models as models

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import lmdb
import time
import tqdm
import os
import pickle
import sys
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset
sys.path.append("../")
from model import AlexNet, resnet18


def get_item_feature(model='resnet18', batch_size=512, device=torch.device("cuda:3")):

    res_path = "./items_features_ori_"+model+".pkl"


    if model == 'alexnet':
        pretrained_path = "/home/clq/datas/pretrained/alexnet-owt-4df8aa71.pth"
        net = AlexNet(num_classes=3)

        my_dict = net.state_dict()
        model_pretrained = torch.load(pretrained_path)

        pretrained_dict = {k: v for k, v in model_pretrained.items() if
                           k != "classifier.6.weight" and k != "classifier.6.bias"}
        my_dict.update(pretrained_dict)

        net.load_state_dict(my_dict)
        net.to(device)
    else:
        pretrained_path = "/home/clq/datas/pretrained/resnet18-5c106cde.pth"
        net = resnet18()

        my_dict = net.state_dict()
        model_pretrained = torch.load(pretrained_path)

        pretrained_dict = {k: v for k, v in model_pretrained.items() if
                           k != "fc.weight" and k != "fc.bias"}
        my_dict.update(pretrained_dict)

        net.load_state_dict(my_dict)
        net.to(device)


    item_feature = dict()
    factor_feature = dict()
    net.eval()

    class MFGN_Dataset(tud.Dataset):
        def __init__(self,):
            super(MFGN_Dataset, self).__init__()
            self.items = []
            for root, dirs, files in os.walk(r"/home/clq/datas/polyvore_outfits/images/"):
                for file in files:
                    # 获取文件路径
                    # print(os.path.join(root, file))
                    self.items.append(os.path.join(root, file))

        def __len__(self):
            return len(self.items)  # 返回数据集中有多少个outfit

        def __getitem__(self, idx):
            return self.items[idx]

    class MFGN_Dataloader(object):
        def __init__(self, batch_size):
            self.dataset = MFGN_Dataset()
            self.loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                num_workers=8,
                shuffle=True,
                pin_memory=False,
            )

        def load_image(self, path):
            """PIL loader for loading image.

            Return
            ------
            img: the image of n-th item in c-the category, type of PIL.Image.
            """
            # read from raw image
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
            return transforms.ToTensor()(img)

        def __iter__(self):
            """Return generator."""
            for img_path in self.loader:
                img = list(map(self.load_image, img_path))
                img = torch.tensor([item.detach().numpy() for item in img])
                img_idx = torch.tensor([int(path.split('/')[-1].split('.')[0]) for path in img_path])
                yield img, img_idx

        def __len__(self):
            """Return number of batches."""
            return len(self.loader)

    train_loader = MFGN_Dataloader(batch_size)
    with torch.no_grad():
        for step, data in enumerate(train_loader):  # 遍历训练集
            time_start = time.perf_counter()
            images, images_idx = data
            outputs = net(images.to(device))  # 正向传播
            outputs = np.array(outputs.cpu())
            for idx, feature in zip(images_idx, outputs):
                # 提取item feature
                # feature = np.array(feature)
                item_feature[idx.item()] = feature
                """
                # 提取因子 feature
                factors = np.linalg.svd(feature)[:10]
                factor_feature[idx] = factors
                """
            print(step, '/', len(train_loader), (time.perf_counter() - time_start)*10, 's')
    with open(res_path, 'wb') as f:
        pickle.dump(item_feature, f)


get_item_feature(model='resnet18', batch_size=512, device=torch.device("cuda:3"))