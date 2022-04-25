import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle
import math
import time
import torch.nn.functional as F


class MFGN(torch.nn.Module):
    def __init__(self, args):
        super(MFGN, self).__init__()
        self.dim = args.dim
        self.device = args.device
        """
        item_features = "./pretrain/items_features.pkl"
        # 读取预处理item特征 1024
        with open(item_features, 'rb') as f:
            fs = pickle.load(f)
        data = torch.zeros(len(fs.keys()), self.dim)
        k = 0
        for v in fs.values():
            data[k] = torch.from_numpy(v)
        """
        # self.item_neighbors_posi = Mydata.train_posi_items_neighbor
        # self.item_neighbors_nega = Mydata.train_nega_items_neighbor
        # self.outfit_items_posi = Mydata.train_posi_outfits_items
        # self.outfit_items_nega = Mydata.train_nega_outfits_items

        self.n_factors = args.n_factors
        self.n_items = args.n_items
        self.n_outfits = args.n_outfits

        # self.outfits_embedding = nn.Embedding(n_outfits, 1024)
        # item->factors
        self.creat_factors_linears = [
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.dim, self.dim//2),
                nn.LeakyReLU(),
                nn.Linear(self.dim//2, self.dim),
                nn.LeakyReLU(),
            ).to(self.device) for _ in range(self.n_factors)]
        # factors->factors
        self.f2f = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            )


        # gate
        self.gate = nn.Sequential(
            nn.Linear(self.dim, 1),
            nn.Softmax(dim=2)
        )

        # factors->item
        self.f2i = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
        )

        # item->item
        self.i2i = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
        )

        # item->outfit
        self.i2o = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
        )

        # outfit->score
        self.o2s = nn.Sequential(
            nn.Linear(self.dim, 1),
            nn.Sigmoid()
        )

        # outfit->score

        # self.items_feature = data
        # self.outfits_feature = self.outfits_embedding(self.n_outfits, 1024)  # 用id初始化outfit

        # print("Init factors")
        # self.creat_factors()  # 初始化item的factors

    def creat_factors(self, items_feature, items_factors, plus):
        for i in range(self.n_outfits):
            for k in range(len(items_feature[i])):
                for j in range(self.n_factors):
                    items_factors[i+plus][k][j] = self.creat_factors_linears[j](items_feature[i+plus][k].clone())
        return items_factors

    # factors进行交互
    def inter_factors(self, outfit_items, items_neighbor, items_factors, plus):
        for k in range(self.n_outfits):  # 遍历这个batch中所有的item
            o = outfit_items[k]
            for i in o:  # 遍历套装i中所有item
                if i.item() == -1:
                    continue
                for j in range(self.n_factors):  # 遍历所有outfit中的item的所有因子
                    t = items_factors[k+plus][i][j].clone()
                    for n in torch.cat((o, items_neighbor[k][i])):  # 求关系
                        if n == -1 or n == i:
                            continue
                        items_factors[k+plus][i][j] = torch.add(items_factors[k+plus][i][j], self.f2f(t*items_factors[k+plus][n][j].clone()))
        return items_factors


    # f2i
    def infer_items(self, outfit_items, items_feature, items_factors, plus):
        for k in range(self.n_outfits):
            o = outfit_items[k]
            for i in o:
                if i.item() == -1:
                    continue
                for j in range(self.n_factors):  # 遍历所有outfit中的item的所有因子
                    items_feature[k][i] = self.f2i(items_factors[k + plus][i][j].clone()) + items_feature[k][i]
        return items_feature

    def inter_items(self, outfit_items, items_feature, plus):
        for k in range(self.n_outfits):
            o = outfit_items[k]
            for i in o:
                if i == -1:
                    continue
                t = items_feature[k][i].clone().data
                for j in o:
                    if j == -1 or j == i:
                        continue
                    t = torch.add(self.i2i(items_feature[k][j]), t)
                items_feature[k][i] = t
        return items_feature

    def infer_outfit(self, outfit_items, items_feature, plus):
        outfits_feature = torch.zeros(len(outfit_items), self.dim).to(self.device)
        for i in range(self.n_outfits):
            o = outfit_items[i]
            for j in o:
                if j == -1:
                    continue
                outfits_feature[i] = torch.add(outfits_feature[i], items_feature[i][j])
        return outfits_feature

    def bpr_loss(self, diff):
        return -torch.mean(torch.log(nn.Sigmoid()(diff+1e-6)))


    def com_loss(self, outfit_items_posi, items_factors):
        E = torch.eye(self.n_factors).to(self.device)
        loss = torch.zeros(1).to(self.device)

        for o in range(self.n_outfits):
            for i in range(self.n_items):
                if outfit_items_posi[o][i] == -1:
                    continue
                loss = loss + (torch.norm(torch.mm(items_factors[o][i], items_factors[o][i].t()) - E) ** 2)

        return loss

    def soft_margin_loss(self, x):
        target = torch.ones_like(x)
        return F.soft_margin_loss(x, target, reduction="none")

    # 输入 items的factors矩阵 item的向量  item相邻的结点的矩阵
    def forward(self, outfit_items_posi, items_feature_posi,  items_neighbor_posi, items_factors):

        self.creat_factors(items_feature_posi, items_factors, 0)

        self.inter_factors(outfit_items_posi, items_neighbor_posi, items_factors, 0)

        # gate = self.gate(items_factors)

        # items_factors = gate * items_factors

        self.infer_items(outfit_items_posi, items_feature_posi, items_factors, 0)

        self.inter_items(outfit_items_posi, items_feature_posi, 0)

        outfits_feature_posi = self.infer_outfit(outfit_items_posi, items_feature_posi, 0)

        scores = self.o2s(outfits_feature_posi).squeeze(1)

        com_loss = self.com_loss(outfit_items_posi, items_factors) / self.n_outfits

        return scores, com_loss

