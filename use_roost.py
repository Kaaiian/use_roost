import os
import gc
import datetime

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score

from roost.message import CompositionNet, ResidualNetwork, \
                            SimpleNetwork
from roost.data import input_parser, CompositionData, \
                        Normalizer, collate_batch
from roost.utils import evaluate, save_checkpoint, \
                        load_previous_state, RobustL1, \
                        RobustL2, cyclical_lr, \
                        LRFinder
from use_train import ensemble, test_ensemble

import math

# %%


def train_roost(args, model_name, csv_train, csv_val=None, val_frac=0.15):

    args.data_path = f'data/datasets/{csv_train}'
    args.fea_path = "data/embeddings/onehot-embedding.json"
    args.val_size = val_frac

    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)
    orig_atom_fea_len = dataset.atom_fea_dim
    args.fea_len = orig_atom_fea_len

    if csv_val is None:
        indices = list(range(len(dataset)))
        train_idx, val_idx = split(indices, random_state=args.seed,
                                   test_size=args.val_size)
        train_set = torch.utils.data.Subset(dataset, train_idx[0::args.sample])
        val_set = torch.utils.data.Subset(dataset, val_idx)
    else:
        train_set = dataset
        val_set = CompositionData(data_path=f'data/datasets/{csv_val}',
                                  fea_path=args.fea_path)

    if not os.path.isdir("models/"):
        os.makedirs("models/")

    if not os.path.isdir("runs/"):
        os.makedirs("runs/")

    if not os.path.isdir("results/"):
        os.makedirs("results/")

    ensemble(model_name, args.fold_id, train_set, val_set,
             args.ensemble, orig_atom_fea_len, args)


def predict_roost(args, model_name, csv_pred):
    fold_id = args.fold_id
    ensemble_folds = args.ensemble
    fea_len = args.fea_len
    args.data_path = f'data/datasets/{csv_pred}'
    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)
    hold_out_set = dataset
    test_ensemble(model_name, fold_id, ensemble_folds,
                  hold_out_set, fea_len, args)


def run():
    args = input_parser()
    args.epochs = 100
    model_name = 'test_name'
    csv_train = 'shear_train.csv'
    csv_val = 'shear_val.csv'
    csv_test = 'shear_test.csv'
    # train_roost(args, model_name, csv_train, csv_val=None, val_frac=0.15)
    train_roost(args, model_name, csv_train, csv_val=csv_val)
    predict_roost(args, model_name, csv_test)


if __name__ == '__main__':
    run()
