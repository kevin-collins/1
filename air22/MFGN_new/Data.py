import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset

import json
import os
from torchvision.transforms import transforms
from PIL import Image
import pickle

import random as rd
import scipy.sparse as sp
from time import time
import math
from tqdm import tqdm
from collections import Counter, defaultdict


class Data(object):
    def __init__(self, idx_path, file_type, max_item_neighbor, split='disjoint', max_outfits=-1):
        self.idx_path = idx_path
        self.max_item = 16 if split == 'disjoint' else 19
        self.max_item_neighbor = max_item_neighbor
        self.max_outfits = max_outfits
        self.train_idx_file = self.idx_path + "/compatibility_train_new." + file_type  # 17316

        self.val_idx_file = self.idx_path + "/compatibility_valid_new." + file_type  # 17316

        self.test_idx_file = self.idx_path + "/compatibility_test_new." + file_type  # 3076

        self.train_outfits_label, self.train_outfits_items, self.train_items_neighbor = \
            self.get_outfits(self.train_idx_file, file_type)

        self.num_outfits = len(self.train_outfits_label)

        self.val_outfits_label, self.val_outfits_items, self.val_items_neighbor = \
            self.get_outfits(self.val_idx_file, file_type)


        # 读取预处理item特征 1024'
        item_features = "F:\projects\MFGN/pretrain/items_features_ori_resnet18.pkl"
        with open(item_features, 'rb') as f:
            fs = pickle.load(f)
        self.items_feature = fs

        if self.max_outfits < self.num_outfits and self.max_outfits != -1:
            self.train_outfits_idx = self.train_outfits_idx[:self.max_outfits]

    def get_outfits(self, path, file_type):
        if file_type == 'txt':
            return self.get_outfits_txt(path)
        elif file_type == 'json':
            return self.get_outfits_json(path)
        else:
            raise ValueError("数据类型格式不是txt或者json！！")

    def get_outfits_txt(self, path):
        outfits_label = []  # 保存outfit的id
        outfits_items = dict()  # 保存outfit对应的item
        items_neighbor = defaultdict(set)  # 保存共现过的item
        with open(path, 'r') as f:
            datas = f.read().split('\n')
            outfits_label = [1 for _ in range(len(datas))]
            o = 0
            for line in datas:
                data = list(map(int, line.split(' ')))
                outfits_label[o] = int(data[0])
                outfits_items[o] = data[1:]

                # 处理相邻结点问题
                for i in outfits_items[o]:
                    for j in outfits_items[o]:
                        if i != j:
                            items_neighbor[i].add(j)
                o += 1

        return outfits_label, outfits_items, items_neighbor

    def get_outfits_json(self, path):
        outfits_label = []  # 保存outfit的id
        outfits_items = dict()  # 保存outfit对应的item
        items_neighbor = defaultdict(set)  # 保存共现过的item
        with open(path, 'r') as f:
            datas = json.load(f)
            outfits_label = [1 for _ in range(len(datas))]
            o = 0
            for outfit in datas:
                outfits_label[o] = int(outfit['label'])
                outfits_items[o] = [i['im'] for i in outfit['items']]

                # 处理相邻结点问题
                for i in outfits_items[o]:
                    for j in outfits_items[o]:
                        if i != j:
                            items_neighbor[i].add(j)
                o += 1

        return outfits_label, outfits_items, items_neighbor


MyData = Data("F:/datas/polyvore_outfits/disjoint", "json", 5)


class MFGN_Train_Dataset(tud.Dataset):
    def __init__(self, data):
        super(MFGN_Train_Dataset, self).__init__()
        self.max_item = data.max_item
        self.max_item_neighbor = data.max_item_neighbor

        self.outfit_label = data.train_outfits_label
        self.outfit_items = data.train_outfits_items
        self.item_neighbors = data.train_items_neighbor

        self.items_feature = data.items_feature

    def __len__(self):
        return len(self.outfit_label)  # 返回数据集中有多少个outfit

    def __getitem__(self, idx):

        # 获取原始outfit中的feature  位置id
        items_feature = [self.items_feature[int(i)] for i in self.outfit_items[idx]]

        # 获取outfit中的item 位置id
        outfit_items = [i for i in range(len(self.outfit_items[idx][:self.max_item]))]

        # 加入相邻item 获取相邻的item的feature，并建立映射表（位置id 映射 位置id） 不包括在当前套装中的
        items_neighbor = [[] for _ in range(len(outfit_items))]
        k = len(items_neighbor)
        for i in range(len(outfit_items)):
            for j in self.item_neighbors[i]:  # 真实id j
                if j not in self.outfit_items[idx]:  # 真实id不在当前套装真实id中
                    items_neighbor[i].append(k)  # 添加j的位置id  k
                    items_feature.append(self.items_feature[j])  # 对应位置添加特征
                    k += 1
            items_neighbor[i] = list(set(items_neighbor[i]))




        # 填补特征长度
        t = self.max_item + self.max_item*self.max_item_neighbor - len(items_feature)
        for i in range(t):
            items_feature.append([0 for _ in range(512)])


        # 填补相邻映射表长度
        t = self.max_item - len(items_neighbor)
        for i in range(t):
            items_neighbor.append([])
        for i in range(len(items_neighbor)):
            items_neighbor[i] = items_neighbor[i][:self.max_item_neighbor]
            t = self.max_item_neighbor - len(items_neighbor[i])
            for j in range(t):
                items_neighbor[i].append(-1)


        # 填补outfit中item的长度
        t = self.max_item - len(self.outfit_items[idx])
        outfit_items = outfit_items + [-1 for _ in range(t)]

        # 类型转换
        outfit_label = torch.tensor(self.outfit_label[idx])

        outfit_items = torch.tensor(outfit_items)

        items_neighbor = torch.tensor(items_neighbor)

        items_feature = torch.tensor(items_feature)



        return outfit_label, outfit_items, items_feature, items_neighbor

class MFGN_Val_Dataset(tud.Dataset):
    def __init__(self, data):
        super(MFGN_Val_Dataset, self).__init__()
        self.max_item = data.max_item
        self.max_item_neighbor = data.max_item_neighbor

        self.outfit_label = data.val_outfits_label
        self.outfit_items = data.val_outfits_items
        self.item_neighbors = data.val_items_neighbor

        self.items_feature = data.items_feature

    def __len__(self):
        return len(self.outfit_label)  # 返回数据集中有多少个outfit

    def __getitem__(self, idx):

        # 获取原始outfit中的feature  位置id
        items_feature = [self.items_feature[int(i)] for i in self.outfit_items[idx]]

        # 获取outfit中的item 位置id
        outfit_items = [i for i in range(len(self.outfit_items[idx][:self.max_item]))]

        # 加入相邻item 获取相邻的item的feature，并建立映射表（位置id 映射 位置id） 不包括在当前套装中的
        items_neighbor = [[] for _ in range(len(outfit_items))]
        k = len(items_neighbor)
        for i in range(len(outfit_items)):
            for j in self.item_neighbors[i]:  # 真实id j
                if j not in self.outfit_items[idx]:  # 真实id不在当前套装真实id中
                    items_neighbor[i].append(k)  # 添加j的位置id  k
                    items_feature.append(self.items_feature[j])  # 对应位置添加特征
                    k += 1
            items_neighbor[i] = list(set(items_neighbor[i]))




        # 填补特征长度
        t = self.max_item + self.max_item*self.max_item_neighbor - len(items_feature)
        for i in range(t):
            items_feature.append([0 for _ in range(512)])


        # 填补相邻映射表长度
        t = self.max_item - len(items_neighbor)
        for i in range(t):
            items_neighbor.append([])
        for i in range(len(items_neighbor)):
            items_neighbor[i] = items_neighbor[i][:self.max_item_neighbor]
            t = self.max_item_neighbor - len(items_neighbor[i])
            for j in range(t):
                items_neighbor[i].append(-1)


        # 填补outfit中item的长度
        t = self.max_item - len(self.outfit_items[idx])
        outfit_items = outfit_items + [-1 for _ in range(t)]

        # 类型转换
        outfit_label = torch.tensor(self.outfit_label[idx])

        outfit_items = torch.tensor(outfit_items)

        items_neighbor = torch.tensor(items_neighbor)

        items_feature = torch.tensor(items_feature)



        return outfit_label, outfit_items, items_feature, items_neighbor