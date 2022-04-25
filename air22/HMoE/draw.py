import time
import scipy.stats
import torch
import os
from options import args_parser
from utils import exp_details
from model import HMoE
from tqdm import tqdm
import sys
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from Data import XDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.decomposition import PCA
pca = PCA(2)

vocabulary_size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4
}

args = args_parser()
args.scenario_nums = 3
exp_details(args)  # 输出参数
args.input_size = len(vocabulary_size) * args.embedding_size


def get_dataloader(filename, batch_size, shuffle):
    data = XDataset(filename)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader




model_path = args.save_main_path
print("model_path", model_path)
model = HMoE(args, vocabulary_size, args.embedding_size)
model.load_state_dict(torch.load(model_path))
model.to(args.device)
print(model)
print("start geting data...")
model.eval()

""" ******************************     数据处理    ************************************* """
# test_dataloader = get_dataloader(args.path_test, args.test_bs, shuffle=True)
# train_dataloader = get_dataloader(args.path_train, args.test_bs, shuffle=True)
dev_dataloader = get_dataloader(args.path_dev, args.test_bs, shuffle=True)
# test_dataloader = dev_dataloader
# train_dataloader = dev_dataloader
for i in range(5):
    # test_click,_, test_features = next(iter(test_dataloader))
    dev_click,_, dev_features = next(iter(dev_dataloader))
    # train_click,_, train_features = next(iter(train_dataloader))

    """
    for key in test_features.keys():
        test_features[key] = test_features[key].to(args.device)
    
    for key in train_features.keys():
        train_features[key] = train_features[key].to(args.device)
    """
    for key in dev_features.keys():
        dev_features[key] = dev_features[key].to(args.device)

    print("projecting")
    # test_features = model(test_features)
    dev_features = model(dev_features)
    # train_features = model(train_features)


    for fs in dev_features:
        for i in range(len(fs)):
            angle = []
            for j in range(len(fs)):
                print(fs[j])
                angle.append(torch.cosine_similarity(fs[i], fs[j], dim=0))
            print(angle)
            exit(0)
    """ ******************************     KNN聚类 看每类里点击的比例    ************************************* """
    """
    data = dev_features.data.cpu()
    label = dev_click.data.cpu()
    # data = torch.cat((test_features, dev_features, train_features), dim=0).data.cpu()
    # label = torch.cat((test_click, dev_click, train_click), dim=0).data.cpu()
    print(data.size())
    print(label.size())
    
    
    model = KMeans(n_clusters=2)
    # 模型拟合
    model.fit(data)
    # 为每个示例分配一个集群
    yhat = model.predict(data)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        click = sum(label[row_ix[0]])
    
        print(click, len(row_ix[0]), click/len(row_ix[0]))
    exit(0)
    """
    """ ******************************     画图    ************************************* """
    print("drawing 数据集分布")

    colors = ['green', 'blue', 'yellow']
    """
    reduced_data_pca = pca.fit_transform(train_features.data.cpu())
    for i, (x, y) in enumerate(reduced_data_pca):
        if train_click[i] == 1:
            type = 0
            plt.scatter(x, y, c=colors[type])

    reduced_data_pca = pca.fit_transform(dev_features.data.cpu())
    for i, (x, y) in enumerate(reduced_data_pca):
        if dev_click[i] == 1:
            type = 1
            plt.scatter(x, y, c=colors[type])

    reduced_data_pca = pca.fit_transform(test_features.data.cpu())
    for i, (x, y) in enumerate(reduced_data_pca):
        if test_click[i] == 1:
            type = 2
            plt.scatter(x, y, c=colors[type])

    # 设置坐标标签
    plt.xlabel(str(i)+'First Principal Component')
    plt.ylabel('Second Principal Component')
    # 设置标题
    plt.title(args.test_model_path)

    # 显示图形
    plt.show()
    """
    """ ******************************     画图    ************************************* """
    """
    reduced_data_pca = pca.fit_transform(train_features.data.cpu())
    for i, (x, y) in enumerate(reduced_data_pca):
        type = 0
        plt.scatter(x, y, c=colors[type])

    reduced_data_pca = pca.fit_transform(dev_features.data.cpu())
    for i, (x, y) in enumerate(reduced_data_pca):
        type = 1
        plt.scatter(x, y, c=colors[type])

    reduced_data_pca = pca.fit_transform(test_features.data.cpu())
    for i, (x, y) in enumerate(reduced_data_pca):
        type = 2
        plt.scatter(x, y, c=colors[type])

    # 设置坐标标签
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    # 设置标题
    plt.title(args.test_model_path)

    # 显示图形
    plt.show()


    for step, batch in tqdm(enumerate(test_dataloader)):
        click, conversion, features = batch
        for key in features.keys():
            features[key] = features[key].to(args.device)

        with torch.no_grad():
            projected = model(features)

        reduced_data_pca = pca.fit_transform(projected.data.cpu())
        colors = ['green', 'blue', 'purple', 'yellow', 'black']

        ans = 0
        n = 0
        # print(reduced_data_pca.shape) [b,2]
        # 根据主成分分析结果绘制散点图
        for i in range(len(features['129'])):
            type = int(features['129'][i])
            x = reduced_data_pca[i][0]
            y = reduced_data_pca[i][1]
            plt.scatter(x, y, c=colors[type])
            if x<0.04:
                n += 1
                ans += int(click[i])
        print(ans, n)

        # 设置坐标标签
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        # 设置标题
        plt.title(args.test_model_path)

        # 显示图形
        plt.show()

    """

