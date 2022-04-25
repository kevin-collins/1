import torch
import torch.nn as nn

import time
import torch
import os
from options import args_parser
from utils import exp_details
import torch
from torch import nn
from torch.utils.data import DataLoader
from Data import XDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from test import test



class BaesDNN(torch.nn.Module):
    def __init__(self, args, feature_vocabulary, embedding_size):
        super(BaesDNN, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.ModuleDict()
        self.device = args.device
        self.__init_weight()

        self.input_size = args.input_size

        self.ctr_DNN = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.cvr_DNN = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            # nn.init.xavier_uniform(emb.weight)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        # 生成特征
        feature_embedding = []
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name])
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1)  # [batch, 90]
        pred_ctr = self.ctr_DNN(feature_embedding)
        pred_cvr = self.cvr_DNN(feature_embedding)
        return torch.squeeze(pred_ctr), torch.squeeze(pred_cvr)  # [batch]