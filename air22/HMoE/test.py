import time

import torch
import os
from options import args_parser
from utils import get_user_groups, average_weights, exp_details, cal_auc, cal_loss, get_model, setup_seed
# from model_AITM import HMoE
from fed_model import HMoE
from tqdm import tqdm
import sys
import random
import torch

from torch.utils.data import DataLoader
from Data import XDataset, get_dataloader, get_dataset, DatasetSplit, vocabulary_size
import numpy as np
from tensorboardX import SummaryWriter

from model import HMoE
from MMoE import MMoE
from ple import PLE
from model_AITM import AITM

setup_seed(2022)

def tast_main(name, path=None):
    print("测试一个模型在全体数据上的结果")

    args = args_parser()
    args.scenario_nums = 3
    exp_details(args)  # 输出参数

    args.input_size = len(vocabulary_size) * args.embedding_size
    if path:
        test_model_path = path
    else:
        test_model_path = args.test_model_path
    print("test_model_path", test_model_path)

    model = get_model(args, name, vocabulary_size)
    model.load_state_dict(torch.load(test_model_path + "main_best.pth"))
    model.to(args.device)

    test_dataloader = get_dataloader(args.path_test, args.test_bs, shuffle=False)

    click_pred = []
    click_label = []
    conversion_pred = []
    conversion_label = []
    for step, batch in tqdm(enumerate(test_dataloader)):
        click, conversion, features = batch
        for key in features.keys():
            features[key] = features[key].to(args.device)

        with torch.no_grad():
            click_prob, conversion_prob, _ = model(features)

        click_pred.append(click_prob.cpu())
        conversion_pred.append(conversion_prob.cpu())

        click_label.append(click)
        conversion_label.append(conversion)

    click_auc = cal_auc(click_label, click_pred)
    conversion_auc = cal_auc(conversion_label, conversion_pred)
    acc = click_auc + conversion_auc
    print(" acc: {} click_auc: {} conversion_auc: {}".format(acc, click_auc, conversion_auc))
    return acc, click_auc, conversion_auc


def tast_fed(name):
    print("用每个客户最好的模型分别测试，结果取平均")
    args = args_parser()
    args.scenario_nums = 3
    exp_details(args)  # 输出参数
    args.input_size = len(vocabulary_size) * args.embedding_size
    test_model_path = args.test_model_path
    print("test_model_path", test_model_path)

    model = get_model(args, name, vocabulary_size)

    test_dataset = get_dataset(args.path_dev)
    _, _, test_idxs, args.num_clients, args.scenario_num = get_user_groups(None, None, test_dataset)
    c = []

    for i in range(5):
        test_dataloader = DataLoader(DatasetSplit(test_dataset, test_idxs[i]),
                                 batch_size=args.test_bs, drop_last=True, shuffle=False)
        print("start test...")
        model.load_state_dict(torch.load(test_model_path+str(i)+"_best.pth"))
        model.to(args.device)
        model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []

        for step, batch in enumerate(test_dataloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(args.device)

            with torch.no_grad():
                click_prob, conversion_prob, _, _ = model(features)

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        acc = click_auc + conversion_auc
        c.append(acc)
        print("client {}, acc: {} click_auc: {} conversion_auc: {}".format(i, acc, click_auc, conversion_auc))
    print(sum(c)/len(c))


def tast_fed_adaptive(name):
    print("用每个客户模型+全局模型 进行测试，结果取平均")
    args = args_parser()
    args.scenario_nums = 3
    exp_details(args)  # 输出参数
    args.input_size = len(vocabulary_size) * args.embedding_size
    test_model_path = args.test_model_path
    print("test_model_path", test_model_path)

    model = get_model(args, name, vocabulary_size)

    print(model)
    global_model = get_model(args, name, vocabulary_size)
    global_model.load_state_dict(torch.load(test_model_path + "main_best.pth"))
    global_model.to(args.device)
    test_dataset = get_dataset(args.path_test)
    _, _, test_idxs, _, _ = get_user_groups(None, None, test_dataset)
    c = []
    alpha = [0.24979, 0.25076, 0.25174, 0.24978, 0.25001]
    for i in range(5):
        test_dataloader = DataLoader(DatasetSplit(test_dataset, test_idxs[i]),
                                 batch_size=args.test_bs, drop_last=True, shuffle=False)
        print("start test...")
        model.load_state_dict(torch.load(test_model_path+str(i)+"_best.pth"))
        model.to(args.device)
        model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []
        for step, batch in tqdm(enumerate(test_dataloader)):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(args.device)

            with torch.no_grad():
                global_click_pred, global_conversion_pred = global_model(features)
                local_click_pred, local_conversion_pred = model(features)
            # 个性化模型结果
            click_prob = alpha[i] * local_click_pred + (1 - alpha[i]) * global_click_pred
            conversion_prob = alpha[i] * local_conversion_pred + (1 - alpha[i]) * global_conversion_pred

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        acc = click_auc + conversion_auc
        c.append(click_auc)
        print("client {}, acc: {} click_auc: {} conversion_auc: {}".format(i, acc, click_auc, conversion_auc))
    print(sum(c)/len(c))


# tast_main("HMoE")
t = {1:'a', 2:'b'}
print(max([t[i] for i in range(len(t.keys()))]))