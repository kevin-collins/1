import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference
from torch.utils.data import DataLoader
from Data import XDataset
from options import args_parser
import torch
from sklearn.metrics import roc_auc_score
from utils import get_user_groups, average_weights, exp_details
from model import HMoE


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

start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('./fed_logs')

args = args_parser()
args.scenario_nums = 3
exp_details(args)  # 输出参数
args.input_size = len(vocabulary_size) * args.embedding_size
if args.parallel:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def get_dataloader(filename, batch_size, shuffle):
    data = XDataset(filename)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_dataset(filename):
    data = XDataset(filename)
    return data


train_dataset = get_dataset(args.path_dev)
test_dataloader = get_dataloader(args.path_dev, args.local_bs, shuffle=False)


# 生成每个client对应的数据 iid
user_groups, args.num_clients, args.scenario_num = get_user_groups(train_dataset)
print("scenario_num is ", args.scenario_num)
print("num_clients is devided by address", args.num_clients)
# args.num_clients = args.scenario_num
# BUILD MODEL
global_model = HMoE(args, vocabulary_size, args.embedding_size)

# Set the model to train and send it to device.
if args.parallel:
    global_model = torch.nn.DataParallel(global_model, device_ids=args.gpu_id)
else:
    global_model.to(args.device)
global_model.train()
print(global_model)

# copy weights
global_weights = global_model.state_dict()

# Training
train_loss, train_accuracy = [], []
train_click_auc, train_conversion_auc = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 1
val_loss_pre, counter = 0, 0
best_acc = 0.0

local_models = []
for i in range(args.num_clients):
    local_models.append(LocalUpdate(args=args, dataset=train_dataset, model=copy.deepcopy(global_model),
                              idxs=user_groups[i], logger=logger))
for epoch in tqdm(range(args.epochs)):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    global_model.train()
    m = max(int(args.frac * args.num_clients), 1)  # 确定数据机个数

    # 取对应的数据机
    idxs_users = np.random.choice(range(args.num_clients), m, replace=False)

    for idx in idxs_users:
        print(f'"start local: {idx} training..."')
        w, loss = local_models[idx].update_weights(id=idx,
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
    list_click_auc, list_conversion_auc = [], []
    global_model.eval()
    # 在客户机上测试全局模型
    print("start test global_model on client")
    for c in tqdm(range(args.num_clients)):
        local_model = local_models[c]
        click_auc, conversion_auc, acc = local_model.inference(id=c, model=global_model, global_round=epoch)
        list_acc.append(acc)
        list_click_auc.append(click_auc)
        list_conversion_auc.append(conversion_auc)
        # torch.save(global_model.state_dict(), args.save_fed_path)

    train_accuracy.append(sum(list_acc)/len(list_acc))
    train_click_auc.append(sum(list_click_auc) / len(list_click_auc))
    train_conversion_auc.append(sum(list_conversion_auc) / len(list_conversion_auc))

    # print global training loss after every 'i' rounds
    if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.3f}%   click_auc: {:.3f}%    conversion_auc {:.3f}% \n'.format(
            100*train_accuracy[-1], 100*train_click_auc[-1], 100*train_conversion_auc[-1]))
    if train_accuracy[-1] > best_acc:
        best_acc = train_accuracy[-1]
        torch.save(global_model.state_dict(), args.save_fed_path)


# 训练结束后进行测试保存工作
# Test inference after completion of training
test_click_auc, test_conversion_auc, test_acc = test_inference(args, global_model, test_dataloader)

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- Avg Train Accuracy: {:.3f}%".format(100*train_accuracy[-1]))
print("|---- Test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%".format(
    100*test_acc, 100*test_click_auc, 100*test_conversion_auc))

"""
# Saving the objects train_loss and train_accuracy:
file_name = '../save/objects/{}_{}_C[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac,
           args.local_ep, args.local_bs)

with open(file_name, 'wb') as f:
    pickle.dump([train_loss, train_accuracy], f)
"""

print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))