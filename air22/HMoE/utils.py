import numpy as np
import torch
import copy
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from tqdm import tqdm
import random

from model import HMoE
from MMoE import MMoE
from ple import PLE
from model_AITM import AITM
# from fed_model import HMoE

# 生成每个client对应的数据 iid
def get_user_groups(train_dataset=None, dev_dataset=None, test_dataset=None):
    train_idxs = defaultdict(set)
    dev_idxs = defaultdict(set)
    test_idxs = defaultdict(set)
    scenario = set()
    if train_dataset:
        for i, (click, conversion, features) in enumerate(train_dataset):
            address = int(features['129'])
            train_idxs[address].add(i)
            scenario.add(int(features['301']))
    if dev_dataset:
        for i, (click, conversion, features) in enumerate(dev_dataset):
            address = int(features['129'])
            dev_idxs[address].add(i)
    if test_dataset:
        for i, (click, conversion, features) in enumerate(test_dataset):
            address = int(features['129'])
            test_idxs[address].add(i)
    return train_idxs, dev_idxs, test_idxs, len(train_idxs), len(scenario)


# 权重求平均
def average_weights(w, train_idxs):
    """
    Returns the average of the weights.
    """
    nums = 0.0
    zb = []
    for idx in train_idxs.values():
        nums += len(idx)
    for idx in train_idxs.values():
        zb.append(len(idx)/nums)
    # print(zb)
    w_avg = copy.deepcopy(w[0][1])
    for key in w_avg.keys():
        w_avg[key] = zb[w[0][0]] * w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] += zb[w[i][0]]*w[i][1][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def alpha_update(global_model, local_model, alpha, eta):
    grad_alpha = 0
    for g_params, l_params in zip(global_model.parameters(), local_model.parameters()):
        if l_params.grad is not None and g_params.grad is not None:
            dif = l_params.data - g_params.data
            grad = alpha * l_params.grad.data + (1 - alpha) * g_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

    grad_alpha += 0.02 * alpha
    alpha_n = alpha - eta * grad_alpha
    alpha_n = np.clip(alpha_n.item(), 0.0, 1.0)
    return alpha_n


def cal_auc(label: list, pred: list):
    label = torch.cat(label)
    pred = torch.cat(pred)
    label = label.detach().numpy()
    pred = pred.detach().numpy()
    auc = roc_auc_score(label, pred, labels=np.array([0.0, 1.0]))
    return auc


def cal_sparse_loss(x):
    return 1.0/(sum(abs(x.view(-1)-torch.tensor(0.5)))+1e-8)


def com_loss(factors, device):
    n, n_factor = factors.size()[0], factors.size()[1]
    t = torch.bmm(factors, torch.transpose(factors, dim0=1, dim1=2)) - \
        torch.unsqueeze(torch.eye(n_factor), dim=0).repeat(n, 1, 1).to(device)
    return torch.norm(t) ** 2


def cal_loss(click_label, click_pred, conversion_label, conversion_pred, gate=None, factors=None, constraint_weight=0.6, device="gpu:0"):
    click_label = click_label.to(device)
    conversion_label = conversion_label.to(device)

    click_loss = nn.BCELoss()(click_pred, click_label)  # 融合了sigmoid和bceloss
    conversion_loss = nn.BCELoss()(conversion_pred, conversion_label)

    label_constraint = torch.maximum(conversion_pred - click_pred,
                                     torch.zeros_like(click_label))
    constraint_loss = torch.sum(label_constraint)

    sparse_loss = 0
    norm_loss = 0
    if gate is not None:
        sparse_loss = cal_sparse_loss(gate) * constraint_weight
    if factors is not None:
        norm_loss = com_loss(factors, device) * constraint_weight
    loss = click_loss + conversion_loss + sparse_loss + norm_loss + constraint_weight * constraint_loss
    return loss

def get_model(args, name, vocabulary_size):
    print("get model: ", name)
    model = None
    if name == 'HMoE':
        model = HMoE(args, vocabulary_size, args.embedding_size)
    elif name == 'MMoE':
        model = MMoE(args.input_size, 1, args.expert_nums, 2, args.device, vocabulary_size)
    elif name == 'PLE':
        model = PLE(num_layers=2, dim_x=args.input_size, dim_experts_out=[256, 128], num_experts=[8, 8, 8], num_tasks=2, dim_tower=[128, 64, 1],
                feature_vocabulary=vocabulary_size, embedding_size=5)
    elif name == 'AITM':
        model = AITM(vocabulary_size, args.embedding_size)
    else:
        print("unknow model...")
        exit(0)
    print(model)
    return model


def splic_model(global_model, local_model, names=None):
    # 保留names中的参数为本地参数
    if names is None:
        return global_model.state_dict()

    model_dict = local_model.state_dict()
    global_state_dict = global_model.state_dict()

    state_dict = dict()
    for k, v in global_state_dict.items():
        if k in names:  # 如果是本地参数
            state_dict[k] = model_dict[k]
        else:  # 如果使用全局参数
            state_dict[k] = v
    model_dict.update(state_dict)
    return model_dict

def test(model, test_dataloader, args):
    print("start test...")
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
            click_prob, conversion_prob, _, _= model(features)

        click_pred.append(click_prob.cpu())
        conversion_pred.append(conversion_prob.cpu())

        click_label.append(click)
        conversion_label.append(conversion)

    # print(set(click_label), set(conversion_label))
    click_auc = cal_auc(click_label, click_pred)
    print(click_auc)
    conversion_auc = cal_auc(conversion_label, conversion_pred)
    acc = click_auc + conversion_auc
    print("test acc: {} click_auc: {} conversion_auc: {}".format(acc, click_auc, conversion_auc))
    return acc, click_auc, conversion_auc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    train on device   : {args.device}\n')
    print(f'    Save Path main   : {args.save_main_path}\n')
    print(f'    Save Path fed  : {args.save_fed_path}\n')

    print(f'    lr  : {args.lr}\n')
    print(f'    weight_decay  : {args.weight_decay}\n')

    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')


    return
