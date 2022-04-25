#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5, help="number of rounds of training")
    parser.add_argument('--note', type=str, default='拆因子 然后过attention聚合', help="train note")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
    parser.add_argument('--device', type=str, default="cuda:3", help="set to a specific GPU ID")
    parser.add_argument('--optimizer', type=str, default='adam', help="type  of optimizer")

    parser.add_argument('--path_train', type=str, default="/home/clq/datas/ali-ccp/ctr_cvr.train")
    parser.add_argument('--path_dev', type=str, default="/home/clq/datas/ali-ccp/ctr_cvr.dev")
    parser.add_argument('--path_test', type=str, default="/home/clq/datas/ali-ccp/ctr_cvr.test")

    parser.add_argument('--save_main_path', type=str, default="/home/clq/projects/HMoE/main_models/")
    parser.add_argument('--save_fed_path', type=str, default="/home/clq/projects/HMoE/fed_models/")

    parser.add_argument('--test_model_path', type=str, default="/home/clq/projects/HMoE/main_models/2022-03-25 14:06:57/")

    parser.add_argument('--fed_log_path', type=str, default="/home/clq/projects/HMoE/fed_log.txt")

    parser.add_argument('--frac', type=float, default=0.6, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=2048, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=4096,  help="test batch size: B")

    parser.add_argument('--parallel', default=False, help="To use gpus parallel")
    parser.add_argument('--gpu', default=True, help=" GPU")
    parser.add_argument('--gpu_id', default=0, help="To use cuda, set  to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--embedding_size', type=int, default=5,  help="embedding_size = 5")
    parser.add_argument('--input_size', type=int, default=90, help="input_size")
    parser.add_argument('--gate_size', type=int, default=64, help="gate_size")
    parser.add_argument('--expert_size', type=int, default=128, help="expert_size")
    parser.add_argument('--fed_expert_size', type=int, default=128, help="expert_size")
    parser.add_argument('--fed_gate_size', type=int, default=64, help="gate_size")

    parser.add_argument('--factor_size', type=int, default=90, help="factor_size")
    parser.add_argument('--factor_nums', type=int, default=3, help="factor_nums")

    parser.add_argument('--expert_nums', type=int, default=5, help="expert_nums")

    parser.add_argument('--tower_size', type=int, default=64, help="tower_size1")
    parser.add_argument('--tower_size2', type=int, default=32, help="tower_size2")

    args = parser.parse_args()
    return args
