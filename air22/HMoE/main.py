import time

import os
from options import args_parser
from utils import exp_details, cal_loss, cal_auc, test, get_model, setup_seed

import torch
from torch import nn
from Data import vocabulary_size, get_dataloader
import numpy as np
from tensorboardX import SummaryWriter

from model import HMoE
from MMoE import MMoE
from ple import PLE
from model_AITM import AITM

setup_seed(2022)
"""          参数处理              """
path_project = os.path.abspath('..')
logger = SummaryWriter('./main_logs')

args = args_parser()
args.scenario_nums = 3  # defaultdict(<class 'int'>, {'2': 14296532, '3': 286913, '1': 23487225})
args.input_size = len(vocabulary_size) * args.embedding_size
exp_details(args)  # 输出参数
print("scenario_num is ", args.scenario_nums)

args.save_main_path = args.save_main_path + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'/'
if not os.path.exists(args.save_main_path):
    os.makedirs(args.save_main_path)
print(args.save_main_path)


"""          模型生成              """
args.model = "HMoE"
model = get_model(args, args.model, vocabulary_size)

model.to(args.device)

args.restore = False
if args.restore:
    print("restore model by ", args.save_main_path)
    pre_train = torch.load(args.save_main_path)
    model.load_state_dict(pre_train)

"""          数据集生成              """

train_dataloader = get_dataloader(args.path_train, args.local_bs, shuffle=True)
dev_dataloader = get_dataloader(args.path_dev, args.local_bs, shuffle=False)
test_dataloader = get_dataloader(args.path_test, args.test_bs, shuffle=False)
"""
dev_dataloader = get_dataloader(args.path_dev, args.local_bs, shuffle=True)
train_dataloader = dev_dataloader
test_dataloader = dev_dataloader
"""


"""          优化器生成              """
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.5)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)


"""          训练              """
best_acc = 0.0
best_ctr_test = 0.0
best_cvr_test = 0.0
best_acc_test = 0.0
best_epoch = 0
for e in range(args.epochs):
    print("start train...")
    model.train()
    t1 = time.time()
    for it, batch in enumerate(train_dataloader):
        click, conversion, features = batch
        for key in features.keys():
            features[key] = features[key].to(args.device)

        click_pred, conversion_pred, gate, factors = model(features)
        loss = cal_loss(click.float(), click_pred, conversion.float(), conversion_pred, gate, factors, device=args.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.add_scalar('main_loss', loss.item(), e * (len(train_dataloader)) + it)
        if it % 100 == 0:
            print("e: %d  %d / %d. loss:%.6f  time: %.3f s" % (e, it, len(train_dataloader), loss.item(), time.time()-t1))
            t1 = time.time()
        if (it+1) % 5000 == 0:
            # validation
            print("start validation...")
            model.eval()
            click_pred = []
            click_label = []
            conversion_pred = []
            conversion_label = []

            for step, batch in enumerate(dev_dataloader):
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

            print("Epoch: {} click_auc: {} conversion_auc: {}".format(
                    e*(len(train_dataloader))+it, click_auc, conversion_auc))
            logger.add_scalar('click_auc', click_auc, e*(len(train_dataloader))+it)
            logger.add_scalar('conversion_auc', conversion_auc, e*(len(train_dataloader))+it)

            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), args.save_main_path+"main_best.pth")

    # test_dataloader = get_dataloader(args.path_test, args.test_bs, shuffle=False)
    acc, click_auc, conversion_auc = test(model, test_dataloader, args)
    best_ctr_test = max(click_auc, best_ctr_test)
    best_cvr_test = max(conversion_auc, best_cvr_test)
    best_acc_test = max(acc, best_acc_test)
    logger.add_scalar('click_auc', click_auc, e)
    logger.add_scalar('conversion_auc', conversion_auc, e)

# 记录本次训练结果和参数
# acc, click_auc, conversion_auc = test(model, test_dataloader, args)
torch.save(model.state_dict(), args.save_main_path + "main_latest.pth")
print("|---- Best test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%  \n".format(
    100*best_acc_test, 100*best_ctr_test, 100*best_cvr_test))
with open("/home/clq/projects/HMoE/main_result/"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+".txt", 'w') as f:
    f.writelines("global model save in " + args.save_main_path+"\n")
    f.writelines(str(model))
    f.writelines("\n\n|---- train_args: lr:{:.8f}   weight_decay:{:.10f}  epoch:{:.0f} \n".format(args.lr, args.weight_decay, args.epochs))
    f.writelines("|---- Best test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%  \n".format(
    100*best_acc_test, 100*best_ctr_test, 100*best_cvr_test))
    f.writelines("factor_nums = {:.0f}".format(args.factor_nums))
    """
    f.writelines("|---- latest epoch Test on Ali-cpp Accuracy: {:.3f}%   click_auc:{:.3f}% conversion_auc:{:.3f}%  \n".format(
    100*acc, 100*click_auc, 100*conversion_auc))
    """
    f.writelines("\n \n \n")