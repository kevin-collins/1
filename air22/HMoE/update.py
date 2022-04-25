#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as tud
import numpy as np
from sklearn import metrics
import time
from utils import alpha_update, cal_auc, cal_loss
from Data import DatasetSplit


class LocalUpdate(object):
    def __init__(self, id, args, model, logger, train_dataset, dev_dataset, test_dataset,
                                train_idxs, dev_idxs, test_idxs):
        self.id = id
        self.args = args
        self.logger = logger
        self.model = model
        self.alpha = 0.2
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(train_dataset, list(train_idxs))
        self.trainloader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                 batch_size=self.args.local_bs, drop_last=True, shuffle=True)
        self.validloader = DataLoader(DatasetSplit(dev_dataset, dev_idxs),
                                 batch_size=self.args.test_bs, drop_last=True, shuffle=False)
        self.testloader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                batch_size=self.args.test_bs, drop_last=True, shuffle=False)
        self.device = 'cuda:3' if args.gpu else 'cpu'
        self.keys = ['bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked']
        # Default criterion set to NLL loss function

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, drop_last=True, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, drop_last=True, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, drop_last=True, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, global_model, global_round):

        """
        # 保留本地的bn参数 其他用global
        model_dict = self.model.state_dict()
        global_state_dict = model.state_dict()
        state_dict = dict()
        for k, v in global_state_dict.items():
            state_dict[k] = 0.5*v + 0.5*model_dict[k]
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)
        """
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               verbose=True,
                                                               threshold=0.00001, threshold_mode='rel', cooldown=0,
                                                               min_lr=1e-6, eps=1e-08)
        print(str(self.id)+" client start train...")
        self.model.train()
        for e in range(self.args.local_ep):
            batch_loss = []
            t1 = time.time()
            for it, batch in enumerate(self.trainloader):
                click, conversion, features = batch
                for key in features.keys():
                    features[key] = features[key].to(self.args.device)
                click_pred, conversion_pred, gate_choose, factors = self.model(features)
                # print(click_pred, conversion_pred, click_pred.size())
                loss = cal_loss(click.float(), click_pred, conversion.float(), conversion_pred, gate_choose, factors, device=self.args.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if it % 50 == 0:
                    print("%d / %d. loss:%.6f  time: %.3f " % (it, len(self.trainloader), loss.item(), time.time()-t1))
                    t1 = time.time()
                    # scheduler.step(loss)
                self.logger.add_scalar(str(id)+'_loss', loss.item(), global_round*e*(len(self.trainloader))+it)
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_agg(self, models, global_round):

        epoch_loss = []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        print(str(self.id)+" client start train...")
        self.model.train()
        for e in range(self.args.local_ep):
            batch_loss = []
            t1 = time.time()
            for it, batch in enumerate(self.trainloader):
                click, conversion, features = batch
                for key in features.keys():
                    features[key] = features[key].to(self.args.device)
                ctrs = []
                cvrs = []
                with torch.no_grad():
                    for model in models:
                        click_pred, conversion_pred, gate_choose, factors = model(features)
                        ctrs.append(click_pred)
                        cvrs.append(conversion_pred)
                click_pred, conversion_pred = self.model(features, torch.stack(ctrs, dim=1).to(self.device), torch.stack(cvrs, dim=1).to(self.device))
                # print(click_pred, conversion_pred, click_pred.size())
                loss = cal_loss(click.float(), click_pred, conversion.float(), conversion_pred, None, None, device=self.args.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if it % 50 == 0:
                    print("%d / %d. loss:%.6f  time: %.3f " % (it, len(self.trainloader), loss.item(), time.time()-t1))
                    t1 = time.time()
                    # scheduler.step(loss)
                self.logger.add_scalar(str(id)+'_loss', loss.item(), global_round*e*(len(self.trainloader))+it)
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_adaptive(self, model, global_round):

        epoch_loss = []
        # Set optimizer for the local updates
        global_optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               verbose=True,
                                                               threshold=0.00001, threshold_mode='rel', cooldown=0,
                                                               min_lr=1e-6, eps=1e-08)
        print(str(self.id)+" client start train...")
        for e in range(self.args.local_ep):
            batch_loss = []
            t1 = time.time()
            for it, batch in enumerate(self.trainloader):
                click, conversion, features = batch
                for key in features.keys():
                    features[key] = features[key].to(self.args.device)

                model.train()
                # 训练全球模型  更新model
                global_click_pred, global_conversion_pred = model(features)
                # print(click_pred, conversion_pred, click_pred.size())
                loss = cal_loss(click.float(), global_click_pred, conversion.float(), global_conversion_pred, device=self.args.device)
                global_optimizer.zero_grad()
                loss.backward()
                global_optimizer.step()

                model.eval()
                self.model.train()
                # 训练本地模型 更新self.model
                global_click_pred, global_conversion_pred = model(features)
                local_click_pred, local_conversion_pred = self.model(features)
                # 个性化模型结果
                click_pred = self.alpha*local_click_pred + (1-self.alpha)*global_click_pred
                conversion_pred = self.alpha*local_conversion_pred + (1-self.alpha)*global_conversion_pred
                loss = cal_loss(click.float(), click_pred, conversion.float(), conversion_pred,device=self.args.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 结果输出
                if it % 50 == 0:
                    print("%d / %d. loss:%.6f  time: %.3f " % (it, len(self.trainloader), loss.item(), time.time()-t1))
                    t1 = time.time()
                    # scheduler.step(loss)
                self.logger.add_scalar(str(id)+'_loss', loss.item(), global_round*e*(len(self.trainloader))+it)
                batch_loss.append(loss.item())
                # 每一次通信只改变一次alpha值
                if e == 0 and it == 0:
                    self.alpha = alpha_update(model, self.model, self.alpha, optimizer.state_dict()['param_groups'][0]['lr'])
                    print(str(self.id)+" client use alpha"+str(self.alpha))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference_adaptive(self, global_model, global_round):
        """ Returns the inference accuracy and loss.
        """
        print(str(self.id)+" client start a*local+(1-a)*global validation...")
        global_model.eval()
        self.model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []

        for step, batch in enumerate(self.validloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            with torch.no_grad():
                global_click_pred, global_conversion_pred = global_model(features)
                local_click_pred, local_conversion_pred = self.model(features)
                click_prob = self.alpha * local_click_pred + (1 - self.alpha) * global_click_pred
                conversion_prob = self.alpha * local_conversion_pred + (1 - self.alpha) * global_conversion_pred

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        print("vaild result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        self.logger.add_scalar(str(self.id)+'_click_auc', click_auc, global_round)
        self.logger.add_scalar(str(self.id) + '_conversion_auc', conversion_auc, global_round)
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc

    def inference_local(self, global_round):
        """ Returns the inference accuracy and loss.
        """
        print(str(self.id)+" client start local validation...")
        self.model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []
        for step, batch in enumerate(self.validloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            with torch.no_grad():
                click_prob, conversion_prob, _, _ = self.model(features)

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        print("vaild result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        self.logger.add_scalar(str(self.id)+'_click_auc', click_auc, global_round)
        self.logger.add_scalar(str(self.id) + '_conversion_auc', conversion_auc, global_round)
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc

    def inference_agg(self, models, global_round):
        """ Returns the inference accuracy and loss.
        """
        print(str(self.id)+" client start local validation...")
        self.model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []
        for step, batch in enumerate(self.validloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            ctrs = []
            cvrs = []
            with torch.no_grad():
                for model in models:
                    click_prob, conversion_prob, gate_choose, factors = model(features)
                    ctrs.append(click_prob)
                    cvrs.append(conversion_prob)
            click_prob, conversion_prob = self.model(features, torch.stack(ctrs, dim=1).to(self.device),
                                                     torch.stack(cvrs, dim=1).to(self.device))
            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        print("vaild result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        self.logger.add_scalar(str(self.id)+'_click_auc', click_auc, global_round)
        self.logger.add_scalar(str(self.id) + '_conversion_auc', conversion_auc, global_round)
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc

    def inference_global(self, global_model, global_round):
        """ Returns the inference accuracy and loss.
        """
        print(str(self.id)+" client start global_model valid...")
        global_model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []

        for step, batch in enumerate(self.validloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            with torch.no_grad():
                click_prob, conversion_prob = global_model(features)

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        print("vaild result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        self.logger.add_scalar(str(self.id)+'_click_auc', click_auc, global_round)
        self.logger.add_scalar(str(self.id) + '_conversion_auc', conversion_auc, global_round)
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc

    # 测试
    def test_adaptive(self, model):
        """ Returns the inference accuracy and loss.
        """
        print(str(self.id)+" client start a*local+(1-a)*global test...")
        model.eval()
        self.model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []

        for step, batch in enumerate(self.testloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            with torch.no_grad():
                global_click_pred, global_conversion_pred = model(features)
                local_click_pred, local_conversion_pred = self.model(features)
                click_prob = self.alpha * local_click_pred + (1 - self.alpha) * global_click_pred
                conversion_prob = self.alpha * local_conversion_pred + (1 - self.alpha) * global_conversion_pred

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        # print("Test result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc

    def test_local(self):
        """ Returns the inference accuracy and loss.
        """
        print(str(self.id)+" client start local test...")
        self.model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []

        for step, batch in enumerate(self.testloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            with torch.no_grad():
                click_prob, conversion_prob, _, _ = self.model(features)

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        # print("Test result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc


    def test_agg(self, models):
        """ Returns the inference accuracy and loss.
        """
        print(str(self.id)+" client start local test...")
        self.model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []

        for step, batch in enumerate(self.testloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            ctrs = []
            cvrs = []
            with torch.no_grad():
                for model in models:
                    click_prob, conversion_prob, gate_choose, factors = model(features)
                    ctrs.append(click_prob)
                    cvrs.append(conversion_prob)
                    click_prob, conversion_prob = self.model(features, torch.stack(ctrs, dim=1).to(self.device),
                                                             torch.stack(cvrs, dim=1).to(self.device))
            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        # print("Test result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc
    # 保存
    def save(self, path, t):
        torch.save(self.model.state_dict(), path + str(self.id) + "_"+t+".pth")
    """
    def test_division(self, global_model):
        print(str(self.id) + " client start division test...")

        # 保留本地的bn参数 其他用global
        model_dict = self.model.state_dict()
        global_state_dict = model.state_dict()
        state_dict = dict()
        for k, v in global_state_dict.items():
            state_dict[k] = 0.5*v + 0.5*model_dict[k]
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)

        self.model.eval()
        click_pred = []
        click_label = []
        conversion_pred = []
        conversion_label = []

        for step, batch in enumerate(self.testloader):
            click, conversion, features = batch
            for key in features.keys():
                features[key] = features[key].to(self.args.device)

            with torch.no_grad():
                click_prob, conversion_prob = self.model(features)

            click_pred.append(click_prob.cpu())
            conversion_pred.append(conversion_prob.cpu())

            click_label.append(click)
            conversion_label.append(conversion)

        # print(set(click_label), set(conversion_label))
        click_auc = cal_auc(click_label, click_pred)
        conversion_auc = cal_auc(conversion_label, conversion_pred)
        # print("Test result  click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))
        acc = click_auc + conversion_auc
        return click_auc, conversion_auc, acc
    """
def test_inference(args, model, test_dataset):
    print("start test...")
    model.eval()
    click_pred = []
    click_label = []
    conversion_pred = []
    conversion_label = []

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_bs, drop_last=True, shuffle=False)

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

    # print(set(click_label), set(conversion_label))
    click_auc = cal_auc(click_label, click_pred)
    conversion_auc = cal_auc(conversion_label, conversion_pred)
    print("click_auc: {} conversion_auc: {}".format(click_auc, conversion_auc))

    acc = click_auc + conversion_auc
    return click_auc, conversion_auc, acc
