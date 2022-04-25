import torch
import torchvision
import numpy as np
import torch.utils.data as tud
from torch import optim
import torch.nn as nn

import pickle
import yaml
import time
import os
from Data import Data, MFGN_Train_Dataset, MFGN_Val_Dataset
from model import MFGN
from sklearn import metrics
from options import args_parser
from utils import get_user_groups, average_weights, exp_details
"""
# 导入用于记录训练过程的日志类
# from logger import Logger

train_log_path = r"./log/train_log"
train_logger = Logger(train_log_path)
# 验证阶段日志文件存储路径
val_log_path = r"./log/val_log"
val_logger = Logger(val_log_path)
"""
idx_path = "/home/clq/datas/datas/polyvore_outfits\disjoint"
img_path = "/home/clq/datas/polyvore_outfits/images/"
config_path = "/home/clq/projects/MFGN/config.yaml"

"""
with open('./read_yaml.yaml', 'r') as f:
    config = yaml.load(f.read())
"""
batch_size = 32
epoch = 15
split = 'disjoint'

args = args_parser()
exp_details(args)  # 输出参数

args.n_outfits = args.batch_size
args.n_items = 16 if split == 'disjoint' else 19
args.device = torch.device("cuda:0")

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# 读取idx信息 建立dataloader
print("build data")

Mydata = Data(args.data_path_polyvore, args.file_type, args.n_neighbor, split='disjoint')
train_dataset = MFGN_Train_Dataset(Mydata)
train_dataloader = tud.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

val_dataset = MFGN_Val_Dataset(Mydata)
val_dataloader = tud.DataLoader(val_dataset, batch_size=64, drop_last=True, shuffle=True)


print("Building model")
model = MFGN(args)

optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)

"""
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model
else:
    model.to(device)
"""
model.to(args.device)

def test(save_path):
    model.load_state_dict(torch.load(save_path, map_location="cuda:0"))
    model.eval()
    val_loss = 0.0
    val_bpr_loss = 0.0
    val_com_loss = 0.0
    val_auc = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            it_start = time.perf_counter()
            label, outfit_items, items_feature, items_neighbor = data
            items_factors = torch.randn(
                [args.n_outfits, args.n_items + args.n_items * args.n_neighbor,
                 args.n_factors, 512])
            optimizer.zero_grad()
            pred, com_loss = model(outfit_items.to(args.device), items_feature.to(args.device),
                                   items_neighbor.to(args.device), items_factors.to(args.device))
            print("froward_time: ", time.perf_counter() - it_start)
            pred = pred.to(torch.float32)
            label = label.to(torch.float32)
            criterion = nn.BCELoss()
            loss = criterion(pred, label) + com_loss * 0.1
            pred = pred.cpu().data.numpy()
            label = label.cpu().data.numpy()
            try:
                auc = metrics.roc_auc_score(label, pred)
            except ValueError:
                pass
            predicts = np.where(pred > 0.5, 1, 0)
            accuracy = metrics.accuracy_score(predicts, label)
            val_loss += loss.item()
            val_com_loss += com_loss.item()
            val_auc += auc.item()
    print("val_loss %.3f    val_com_loss %.3f  val_auc %.3f " % (
        val_loss, val_com_loss, val_auc / float(len(val_dataloader))))


def train(save_path):
    for e in range(epoch):
        model.train()
        e_loss = 0.0
        e_bpr_loss = 0.0
        e_com_loss = 0.0
        e_auc = 0.0
        e_acc = 0.0
        best_auc = 0
        for i, data in enumerate(train_dataloader):
            it_start = time.perf_counter()
            label, outfit_items, items_feature, items_neighbor = data
            items_factors = torch.randn(
                [args.n_outfits, args.n_items + args.n_items * args.n_neighbor,
                 args.n_factors, 512])
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            pred, com_loss =  pred, com_loss = model(outfit_items.to(args.device), items_feature.to(args.device),
                                   items_neighbor.to(args.device), items_factors.to(args.device))
            print("froward_time: ", time.perf_counter() - it_start)
            pred = pred.to(torch.float32)
            label = label.to(torch.float32)
            criterion = nn.BCELoss()
            # print(pred, label)
            loss = criterion(pred, label.to(args.device)) + com_loss * 0.1

            loss.backward()

            pred = pred.cpu().data.numpy()
            label = label.cpu().data.numpy()
            auc = 0
            try:
                auc = metrics.roc_auc_score(label, pred)
            except ValueError:
                pass
            predicts = np.where(pred > 0.5, 1, 0)
            accuracy = metrics.accuracy_score(predicts, label)

            e_loss += loss.item()
            e_com_loss += com_loss.item()
            e_auc += auc
            e_acc += accuracy


            print("epoch %d [%d / %d]  loss %.3f    com_loss %.3f  auc %.3f  acc %.3f time  %.3f s" % (
                e, i, len(train_dataloader), loss.item(),  com_loss.item(), auc, accuracy, (time.perf_counter() - it_start)))
            optimizer.step()

        print("epoch %d  train_loss %.3f   train_com_loss %.3f  avg_train_auc %.3f avg_train_acc %.3f " % (
            e, e_loss, e_com_loss, e_auc / float(len(train_dataloader)), e_acc / float(len(train_dataloader))))
        """
        train_logger.scalar_summary("total loss", e_loss, e)
        train_logger.scalar_summary("bpr loss", e_bpr_loss, e)
        train_logger.scalar_summary("com loss", e_com_loss, e)
        train_logger.scalar_summary("acc", e_acc / float(len(train_dataloader)), e)
        """
        model.eval()
        val_loss = 0.0
        val_com_loss = 0.0
        val_auc = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                it_start = time.perf_counter()
                label, outfit_items, items_feature, items_neighbor = data
                items_factors = torch.randn(
                    [args.n_outfits, args.n_items + args.n_items * args.n_neighbor,
                     args.n_factors, 512])
                optimizer.zero_grad()
                pred, com_loss = model(outfit_items.to(args.device), items_feature.to(args.device),
                                       items_neighbor.to(args.device), items_factors.to(args.device))
                print("froward_time: ", time.perf_counter() - it_start)
                pred = pred.to(torch.float32)
                label = label.to(torch.float32)
                criterion = nn.BCELoss()
                loss = criterion(pred, label.to(args.device)) + com_loss * 0.1
                pred = pred.cpu().data.numpy()
                label = label.cpu().data.numpy()
                try:
                    auc = metrics.roc_auc_score(label, pred)
                except ValueError:
                    pass
                predicts = np.where(pred > 0.5, 1, 0)
                accuracy = metrics.accuracy_score(predicts, label)

                val_loss += loss.item()
                val_com_loss += com_loss.item()
                val_auc += auc
                val_acc += accuracy
        print("epoch %d  val_loss %.3f  val_com_loss %.3f  val_auc %.3f  val_acc %.3f " % (
            e, val_loss, val_com_loss, val_auc / float(len(val_dataloader)), val_acc / float(len(val_dataloader))))

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)


train("./model_6_pb.pth")