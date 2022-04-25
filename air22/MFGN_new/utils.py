import numpy as np
import torch
import copy

# 生成每个client对应的数据 iid
def get_user_groups(dataset, num_clients):
    num_items = int(len(dataset)/num_clients)
    user_groups, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        user_groups[i] = set(np.random.choice(all_idxs, num_items,replace=False))
        all_idxs = list(set(all_idxs) - user_groups[i])
    return user_groups


# 权重求平均
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
