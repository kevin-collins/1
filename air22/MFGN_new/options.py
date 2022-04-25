#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.01,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--num_polyvore_clients', type=int, default=100,
                        help="num_polyvore_clients")
    parser.add_argument('--num_pog_clients', type=int, default=100,
                        help="num_pog_clients")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    parser.add_argument('--n_factors', type=int, default=3, help='nums of factor')

    """  *****************       batch_size        *****************        """
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')

    parser.add_argument('--n_items', type=int, default=16, help='n_items')

    parser.add_argument('--n_neighbor', type=int, default=5, help='n_neighbor')

    parser.add_argument('--dim', type=int, default=512, help='size of feature')

    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")

    # data
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                            of dataset")

    parser.add_argument('--file_type', type=str, default='json', help="file type of data")

    parser.add_argument('--data_path_polyvore', type=str, default="F:/datas/polyvore_outfits/disjoint",
                        help="data path of polyvore")

    parser.add_argument('--data_path_pog', type=str, default="F:/datas/polyvore_outfits/disjoint",
                        help="data path of pog")

    # other arguments

    parser.add_argument('--gpu', default=True, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--gpu_id', default=0, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")

    """  *****************   Test    batch_size        *****************        """
    parser.add_argument('--test_batch_size', type=int, default=512, help="test \
                            batch_size")

    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
