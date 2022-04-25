import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from options import args_parser

from update import LocalUpdate, test_inference
from Data import Data, MFGN_Train_Dataset, MFGN_Val_Dataset

from utils import get_user_groups, average_weights, exp_details
from model import MFGN

start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')


args = args_parser()
exp_details(args)  # 输出参数

args.n_outfits = args.batch_size
args.n_items = 16

if args.gpu_id:
    torch.cuda.set_device(args.gpu_id)
device = 'cuda' if args.gpu else 'cpu'

args.device = device

# load dataset and user groups  给出的是所有数据的dataset
data_polyvore = Data(args.data_path_polyvore, args.file_type, args.n_neighbor, split='disjoint')
data_pog = Data(args.data_path_pog, args.file_type, args.n_neighbor, split='disjoint')

train_dataset_polyvore = MFGN_Train_Dataset(data_polyvore)
test_dataset_polyvore = MFGN_Val_Dataset(data_polyvore)

train_dataset_pog = MFGN_Train_Dataset(data_pog)
test_dataset_pog = MFGN_Train_Dataset(data_pog)


# 生成每个client对应的数据 iid
user_groups_polyvore = get_user_groups(train_dataset_polyvore, args.num_polyvore_clients)
user_groups_pog = get_user_groups(train_dataset_pog, args.num_pog_clients)


# BUILD MODEL
global_model = MFGN(args)

# Set the model to train and send it to device.
global_model.to(device)
global_model.train()
print(global_model)

# copy weights
global_weights = global_model.state_dict()

# Training
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 1
val_loss_pre, counter = 0, 0

for epoch in tqdm(range(args.epochs)):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    global_model.train()
    m_polyvore = max(int(args.frac * args.num_polyvore_clients), 1)  # 确定数据机个数
    m_pog = max(int(args.frac * args.num_pog_clients), 1)

    # 取对应的数据机
    idxs_users_polyvore = np.random.choice(range(args.num_polyvore_clients), m_polyvore, replace=False)
    idxs_users_pog = np.random.choice(range(args.num_pog_clients), m_pog, replace=False)

    print("start local polyvore")
    for idx in idxs_users_polyvore:
        local_model = LocalUpdate(args=args, dataset=train_dataset_polyvore,
                                  idxs=user_groups_polyvore[idx], logger=logger)
        w, loss = local_model.update_weights(
            model=copy.deepcopy(global_model), global_round=epoch)
        # 保存客户机更新后的模型权重
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

    print("start local pog")
    for idx in idxs_users_pog:
        local_model = LocalUpdate(args=args, dataset=train_dataset_pog,
                                  idxs=user_groups_pog[idx], logger=logger)
        w, loss = local_model.update_weights(
            model=copy.deepcopy(global_model), global_round=epoch)
        # 保存客户机更新后的模型权重
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

    # update global weights 客户机权重取平均 为下一个阶段的全局模型
    global_weights = average_weights(local_weights)

    # update global weights
    global_model.load_state_dict(global_weights)

    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    # Calculate avg training accuracy over all users at every epoch
    list_acc, list_loss = [], []
    global_model.eval()
    # 在客户机上测试全局模型
    print("start test global_model on client")
    for c in tqdm(range(args.num_polyvore_clients)):
        local_model = LocalUpdate(args=args, dataset=train_dataset_polyvore,
                                  idxs=user_groups_polyvore[c], logger=logger)
        loss, _, auc, acc = local_model.inference(model=global_model)
        list_acc.append(acc)
        list_loss.append(loss)

    train_accuracy.append(sum(list_acc)/len(list_acc))

    # print global training loss after every 'i' rounds
    if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


# 训练结束后进行测试保存工作
# Test inference after completion of training
test_acc_polyvore, test_loss_polyvore = test_inference(args, global_model, test_dataset_polyvore)
test_acc_pog, test_loss_pog = test_inference(args, global_model, test_dataset_pog)

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- Avg Train on Polyvore Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
print("|---- Test on Polyvore Accuracy: {:.2f}%".format(100*test_acc_polyvore))

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- Avg Train on Pog Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
print("|---- Test on Pog Accuracy: {:.2f}%".format(100*test_acc_pog))

"""
# Saving the objects train_loss and train_accuracy:
file_name = '../save/objects/{}_{}_C[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac,
           args.local_ep, args.local_bs)

with open(file_name, 'wb') as f:
    pickle.dump([train_loss, train_accuracy], f)
"""

print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))