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
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from utils import get_user_groups, average_weights, exp_details, cal_auc, cal_loss, get_model, splic_model, setup_seed

setup_seed(2022)


class BaesDNN(torch.nn.Module):
    def __init__(self, args, feature_vocabulary, embedding_size):
        super(BaesDNN, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.ModuleDict()
        self.device = args.device
        self.num_clients = args.num_clients
        self.__init_weight()

        self.input_size = args.input_size

        self.ctr_DNN = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_clients),
            nn.Softmax(),
        )

        self.cvr_DNN = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_clients),
            nn.Softmax(dim=1),
        )

    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            # nn.init.xavier_uniform(emb.weight)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x, ctr, cvr):
        # 生成特征
        feature_embedding = []
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name])
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1)  # [batch, 90]
        gate_ctr = torch.unsqueeze(self.ctr_DNN(feature_embedding), dim=1)  # [batch, 1, num_clients]
        gate_cvr = torch.unsqueeze(self.cvr_DNN(feature_embedding), dim=1)
        pred_ctr = torch.bmm(gate_ctr, torch.unsqueeze(ctr, dim=2))  # [batch, num_clients, 1]
        pred_cvr = torch.bmm(gate_cvr, torch.unsqueeze(cvr, dim=2))
        return torch.squeeze(pred_ctr), torch.squeeze(pred_cvr)  # [batch]


start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('./fed_logs')

args = args_parser()
args.scenario_nums = 3
args.num_clients = 5
exp_details(args)  # 输出参数
args.input_size = len(vocabulary_size) * args.embedding_size
if args.parallel:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# BUILD MODEL  如果要换模型 这里换掉就可以了
args.model = "HMoE"
models = [get_model(args, args.model, vocabulary_size).to(args.device) for _ in range(args.num_clients)]

test_model_path = "/home/clq/projects/HMoE/fed_models/2022-03-30 19:04:29/"
for i in range(args.num_clients):
    models[i].load_state_dict(torch.load(test_model_path+str(i)+"_best.pth"))

DNN = BaesDNN(args, vocabulary_size, args.embedding_size).to(args.device)
# 获取数据集
"""
train_dataset = get_dataset(args.path_train)
dev_dataset = get_dataset(args.path_dev)
test_dataset = get_dataset(args.path_test)
"""
dev_dataset = get_dataset(args.path_dev)
train_dataset = dev_dataset
test_dataset = dev_dataset


# 生成每个client对应的数据 iid
train_idxs, dev_idxs, test_idxs, args.num_clients, args.scenario_num = get_user_groups(train_dataset, dev_dataset,
                                                                                       test_dataset)
print("scenario_num is ", args.scenario_num)
print("num_clients is devided by address", args.num_clients)
# args.num_clients = args.scenario_num


# Training
train_loss, train_accuracy = [], []
train_click_auc, train_conversion_auc = [], []
print_every = 1
best_acc = [0.0] * (args.num_clients + 1)

# 生成各客户机模型
local_models = []
for i in range(args.num_clients):
    local_models.append(LocalUpdate(id=i, args=args, model=copy.deepcopy(DNN), logger=logger,
                                    train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
                                    train_idxs=train_idxs[i], dev_idxs=dev_idxs[i], test_idxs=test_idxs[i]))

# 确定模型保存路径
args.save_fed_path = args.save_fed_path + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '/'
if not os.path.exists(args.save_fed_path):
    os.makedirs(args.save_fed_path)

local_names = {
    "embedding_dict.101.weight",
    "embedding_dict.121.weight",
    "embedding_dict.122.weight",
    "embedding_dict.124.weight",
    "embedding_dict.125.weight",
    "embedding_dict.126.weight",
    "embedding_dict.127.weight",
    "embedding_dict.128.weight",
    "embedding_dict.129.weight",
    "embedding_dict.205.weight",
    "embedding_dict.206.weight",
    "embedding_dict.207.weight",
    "embedding_dict.216.weight",
    "embedding_dict.508.weight",
    "embedding_dict.509.weight",
    "embedding_dict.702.weight",
    "embedding_dict.853.weight",
    "embedding_dict.301.weight",
    "gate_choose.0.weight",
    "gate_choose.0.bias",
    "gate_choose.1.weight",
    "gate_choose.1.bias",
    "attention.q_layer.weight",
    "attention.k_layer.weight",
    "attention.v_layer.weight", }
# 训练
for epoch in tqdm(range(args.epochs)):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch + 1} |\n')


    m = max(int(args.frac * args.num_clients), 1)  # 确定数据机个数
    # 取对应的数据机
    idxs_users = np.random.choice(range(args.num_clients), m, replace=False)

    """    训练      """
    for idx in idxs_users:
        print(f'"start local: {idx} training..."')
        w, loss = local_models[idx].update_weights_agg(models, global_round=epoch)
        # 保存客户机更新后的模型权重
        local_weights.append((idx, copy.deepcopy(w)))
        local_losses.append(copy.deepcopy(loss))


    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    # Calculate avg training accuracy over all users at every epoch
    list_acc, list_click_auc, list_conversion_auc = [], [], []
    """    验证集   更新本地模型   """
    print("start valid local_model on client")
    for c in tqdm(range(args.num_clients)):
        # 进行预测
        click_auc, conversion_auc, acc = local_models[c].inference_agg(models, global_round=epoch)
        list_acc.append(acc)
        list_click_auc.append(click_auc)
        list_conversion_auc.append(conversion_auc)
        if acc > best_acc[c + 1]:
            best_acc[c + 1] = acc
            print("save ", c)
            local_models[c].save(args.save_fed_path, "best")

    train_accuracy.append(sum(list_acc) / len(list_acc))
    train_click_auc.append(sum(list_click_auc) / len(list_click_auc))
    train_conversion_auc.append(sum(list_conversion_auc) / len(list_conversion_auc))

    # print global training loss
    if (epoch + 1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.3f}%   click_auc: {:.3f}%    conversion_auc {:.3f}% \n'.format(
            100 * train_accuracy[-1], 100 * train_click_auc[-1], 100 * train_conversion_auc[-1]))
    if train_click_auc[-1] > best_acc[0]:
        best_acc[0] = train_click_auc[-1]


# 训练结束后进行测试保存工作
for i in range(args.num_clients):
    local_models[i].save(args.save_fed_path, "latest")
print("global model and local models have saved in " + args.save_fed_path)
# Test inference after completion of training
test_acc = 0.0
test_click_auc = 0.0
test_conversion_auc = 0.0

"""    测试      """
for model in local_models:
    click_auc, conversion_auc, acc = model.test_agg(models)
    test_click_auc += click_auc
    test_conversion_auc += conversion_auc
    test_acc += acc

test_acc /= len(local_models)
test_click_auc /= len(local_models)
test_conversion_auc /= len(local_models)

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- max dev Accuracy: {:.3f}%   click_auc:{:.3f}% ".format(100 * max(train_accuracy),
                                                                    100 * max(train_click_auc)))
print("|---- clients test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%".format(
    100 * test_acc, 100 * test_click_auc, 100 * test_conversion_auc))

global_click_auc, global_conversion_auc, global_acc = test_inference(args, global_model, test_dataset)

print("|---- global test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%".format(
    100 * global_acc, 100 * global_click_auc, 100 * global_conversion_auc))
# 记录本次训练结果和参数
with open("/home/clq/projects/HMoE/fed_result/" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".txt",
          'w') as f:
    f.writelines("models save in " + args.save_fed_path + " \n")
    f.writelines(str(global_model))
    f.writelines(
        "\n\n |---- train_args: lr:{:.6f}  weight_decay:{:.10f}  epoch:{:.0f} frac:{:.1f}  local_ep:{:.0f} \n".format(
            args.lr, args.weight_decay, args.epochs, args.frac, args.local_ep))
    f.writelines("|---- max client dev Accuracy: {:.3f}%   click_auc:{:.3f}% \n".format(100 * max(train_accuracy),
                                                                                        100 * max(train_click_auc)))
    f.writelines(
        "|---- clients test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%  \n".format(
            100 * test_acc, 100 * test_click_auc, 100 * test_conversion_auc))
    f.writelines("|---- global test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}% \n".format(
        100 * global_acc, 100 * global_click_auc, 100 * global_conversion_auc))
    f.writelines("|---- alphas is {:.5f},{:.5f},{:.5f},{:.5f},{:.5f} \n".format(
        local_models[0].alpha, local_models[1].alpha, local_models[2].alpha, local_models[3].alpha,
        local_models[4].alpha))
    f.writelines(args.note + "\n")
    f.writelines("\n \n \n")

print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
