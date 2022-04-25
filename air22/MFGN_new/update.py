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


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

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
                                 batch_size=self.args.batch_size, drop_last=True, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.batch_size, drop_last=True, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.batch_size, drop_last=True, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_auc = []
            for batch_idx, data in enumerate(self.trainloader):
                it_start = time.perf_counter()
                label, outfit_items, items_feature, items_neighbor = data
                items_factors = torch.randn(
                    [self.args.n_outfits, self.args.n_items + self.args.n_items * self.args.n_neighbor,
                     self.args.n_factors, 512])
                optimizer.zero_grad()
                pred, com_loss = model(outfit_items.to(self.args.device), items_feature.to(self.args.device),
                                       items_neighbor.to(self.args.device), items_factors.to(self.args.device))

                # print("froward_time: ", time.perf_counter() - it_start)
                pred = pred.to(torch.float32)
                label = label.to(torch.float32)
                criterion = nn.BCELoss()
                # print(pred, label)
                loss = criterion(pred, label.to(self.args.device)) + com_loss * 0.1

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

                # torch.autograd.set_detect_anomaly(True)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}   auc：{:.3f} \
                          time: {:.3f}'.format(
                        global_round, iter, batch_idx*len(label),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item(), auc, time.perf_counter()-it_start))
                self.logger.add_scalar('loss', loss.item())
                self.logger.add_scalar('auc', auc)
                batch_loss.append(loss.item())
                batch_auc.append(auc)
                optimizer.step()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        test_loss = 0.0
        test_com_loss = 0.0
        test_auc = 0.0
        test_acc = 0.0
        total = len(self.testloader)
        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                label, outfit_items, items_feature, items_neighbor = data
                items_factors = torch.randn([self.args.n_outfits, self.args.n_items + self.args.n_items * self.args.n_neighbor,
                                             self.args.n_factors, 512])
                pred, com_loss = model(outfit_items.to(self.args.device), items_feature.to(self.args.device),
                                       items_neighbor.to(self.args.device), items_factors.to(self.args.device))
                pred = pred.to(torch.float32)
                label = label.to(torch.float32)
                criterion = nn.BCELoss()
                loss = criterion(pred, label.to(self.args.device)) + com_loss * 0.1
                pred = pred.cpu().data.numpy()
                label = label.cpu().data.numpy()
                try:
                    auc = metrics.roc_auc_score(label, pred)
                except ValueError:
                    pass
                predicts = np.where(pred > 0.5, 1, 0)
                accuracy = metrics.accuracy_score(predicts, label)
                test_loss += loss.item()
                test_com_loss += com_loss.item()
                test_auc += auc
                test_acc += accuracy
        return test_loss, test_com_loss, test_auc / float(total), test_acc / float(total)


def test_inference(args, model, test_dataset):
    model.eval()
    test_loss = 0.0
    test_com_loss = 0.0
    test_auc = 0.0
    test_acc = 0.0
    val_dataloader = tud.DataLoader(test_dataset, batch_size=args.test_batch_size, drop_last=True)
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            label, outfit_items, items_feature, items_neighbor = data
            items_factors = torch.randn([args.n_outfits, args.n_items + args.n_items * args.n_neighbor,
                 args.n_factors, 512])
            pred, com_loss = model(outfit_items.to(args.device), items_feature.to(args.device),
                                   items_neighbor.to(args.device), items_factors.to(args.device))
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
            test_loss += loss.item()
            test_com_loss += com_loss.item()
            test_auc += auc
            test_acc += accuracy
    return test_loss, test_com_loss, test_auc / float(len(val_dataloader)), test_acc / float(len(val_dataloader))
