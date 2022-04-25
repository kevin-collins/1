import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference
from torch.utils.data import DataLoader
from Data import XDataset, get_dataloader, get_dataset, vocabulary_size
from options import args_parser
import torch
from sklearn.metrics import roc_auc_score
from utils import get_user_groups, average_weights, exp_details
from model import HMoE


start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('./local_logs')

args = args_parser()
args.scenario_nums = 3
exp_details(args)  # 输出参数
args.input_size = len(vocabulary_size) * args.embedding_size
if args.parallel:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


# 获取数据集

train_dataset = get_dataset(args.path_train)
dev_dataset = get_dataset(args.path_dev)
test_dataset = get_dataset(args.path_test)
"""

dev_dataset = get_dataset(args.path_dev)
train_dataset = dev_dataset
test_dataset = dev_dataset
"""

# 生成每个client对应的数据 iid
train_idxs, dev_idxs, test_idxs, args.num_clients, args.scenario_num = get_user_groups(train_dataset, dev_dataset, test_dataset)
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
print_every = 1
best_acc = [0.0]*(args.num_clients+1)

# 生成各客户机模型
local_models = []
for i in range(args.num_clients):
    local_models.append(LocalUpdate(id=i, args=args, model=copy.deepcopy(global_model),logger=logger,
                                    train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
                                    train_idxs=train_idxs[i], dev_idxs=dev_idxs[i], test_idxs=test_idxs[i]))

# 确定模型保存路径
args.save_fed_path = args.save_fed_path + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'/'
if not os.path.exists(args.save_fed_path):
    os.makedirs(args.save_fed_path)


# 训练
for epoch in tqdm(range(args.epochs)):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    m = max(int(args.frac * args.num_clients), 1)  # 确定数据机个数
    # 取对应的数据机
    idxs_users = np.random.choice(range(args.num_clients), m, replace=False)

    """    训练      """
    for idx in idxs_users:
        print(f'"start local: {idx} training..."')
        w, loss = local_models[idx].update_weights(global_model=copy.deepcopy(global_model), global_round=epoch)
        local_losses.append(copy.deepcopy(loss))

    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    # Calculate avg training accuracy over all users at every epoch
    list_acc, list_click_auc, list_conversion_auc = [], [], []
    """    验证集      """
    print("start test global_model on client")
    for c in tqdm(range(args.num_clients)):
        local_model = local_models[c]
        click_auc, conversion_auc, acc = local_model.inference_local(global_round=epoch)
        list_acc.append(acc)
        list_click_auc.append(click_auc)
        list_conversion_auc.append(conversion_auc)
        if acc > best_acc[c+1]:
            best_acc[c+1] = acc
            torch.save(local_model.model.state_dict(), args.save_fed_path+str(c)+"_best.pth")

    train_accuracy.append(sum(list_acc)/len(list_acc))
    train_click_auc.append(sum(list_click_auc) / len(list_click_auc))
    train_conversion_auc.append(sum(list_conversion_auc) / len(list_conversion_auc))

    # print global training loss
    if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.3f}%   click_auc: {:.3f}%    conversion_auc {:.3f}% \n'.format(
            100*train_accuracy[-1], 100*train_click_auc[-1], 100*train_conversion_auc[-1]))


# 训练结束后进行测试保存工作
for i in range(args.num_clients):
    torch.save(local_models[i].model.state_dict(), args.save_fed_path+str(i)+"_latest.pth")
print("global model and local models have saved in "+args.save_fed_path)
# Test inference after completion of training
test_acc = 0.0
test_click_auc = 0.0
test_conversion_auc = 0.0

"""    测试      """
for model in local_models:
    click_auc, conversion_auc, acc = model.test_local()
    test_click_auc += click_auc
    test_conversion_auc += conversion_auc
    test_acc += acc

test_acc /= len(local_models)
test_click_auc /= len(local_models)
test_conversion_auc /= len(local_models)

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- max dev Accuracy: {:.3f}%   click_auc:{:.3f}% ".format(100*max(train_accuracy), 100*max(train_click_auc)))
print("|---- clients test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%".format(
    100*test_acc, 100*test_click_auc, 100*test_conversion_auc))


# 记录本次训练结果和参数
with open("/home/clq/projects/HMoE/fed_result/"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+".txt", 'w') as f:
    f.writelines("models save in "+args.save_fed_path+" \n")
    f.writelines(str(global_model))
    f.writelines("\n\n |---- train_args: lr:{:.6f}  weight_decay:{:.10f}  epoch:{:.0f} frac:{:.1f}  local_ep:{:.0f} \n".format(
        args.lr, args.weight_decay, args.epochs, args.frac, args.local_ep))
    f.writelines("|---- max client dev Accuracy: {:.3f}%   click_auc:{:.3f}% \n".format(100*max(train_accuracy), 100*max(train_click_auc)))
    f.writelines("|---- clients test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%  \n".format(
    100*test_acc, 100*test_click_auc, 100*test_conversion_auc))
    f.writelines(args.note+"\n")
    f.writelines("\n \n \n")


print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))