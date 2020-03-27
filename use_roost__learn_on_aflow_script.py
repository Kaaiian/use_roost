import os
import gc
import datetime

import numpy as np
import pandas as pd
# from tqdm.autonotebook import tqdm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score, mean_absolute_error

from roost.message import CompositionNet, ResidualNetwork, \
                            SimpleNetwork
from roost.data import input_parser, CompositionData, \
                        Normalizer, collate_batch
from roost.utils import evaluate, save_checkpoint, \
                        load_previous_state, RobustL1, \
                        RobustL2, cyclical_lr, \
                        LRFinder, \
                        pva_plot
from use_train import ensemble, test_ensemble

import math

# %%


def train_roost(args, model_name, csv_train, csv_val=None, val_frac=0.0,
                resume=False, transfer=None, fine_tune=None):

    args.data_path = f'data/datasets/{csv_train}'
    args.val_size = val_frac

    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)
    orig_atom_fea_len = dataset.atom_fea_dim
    args.fea_len = orig_atom_fea_len

    if resume:
        args.resume = resume
    else:
        if transfer is not None:
            args.transfer = transfer
        elif fine_tune is not None:
            args.fine_tune = fine_tune

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


def generate_standard_model(mat_prop, device):
    args = input_parser()
    args.device = device
    args.optim = 'AdamW'
    args.epochs = 300
    args.fea_path = "data/embeddings/matscholar-embedding.json"
    model_name = f'{mat_prop}_model_{args.epochs}_epochs'
    csv_train = f'aflow/{mat_prop}/train.csv'
    csv_val = f'aflow/{mat_prop}/val.csv'
    csv_test = f'aflow/{mat_prop}/test.csv'
    # define dataset
    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)
    orig_atom_fea_len = dataset.atom_fea_dim
    args.fea_len = orig_atom_fea_len
    # train and test model
    train_roost(args, model_name, csv_train, csv_val=csv_val)
    predict_roost(args, model_name, csv_test)


def generate_cgcnn_aflow_trained_model(mat_prop, device):
    args = input_parser()
    args.device = device
    args.optim = 'AdamW'
    args.epochs = 250
    args.fea_path = "data/embeddings/matscholar-embedding.json"
    model_name = f'{mat_prop}_cgcnn_aflow_model_{args.epochs}_epochs'
    csv_train = f'cgcnn_aflow/{mat_prop}_cgcnn_pred.csv'
    csv_val = f'aflow/{mat_prop}/val.csv'
    csv_test = f'aflow/{mat_prop}/test.csv'
    # define dataset
    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)
    orig_atom_fea_len = dataset.atom_fea_dim
    args.fea_len = orig_atom_fea_len
    # train and test model
    train_roost(args, model_name, csv_train, csv_val=csv_val)
    predict_roost(args, model_name, csv_test)


def generate_cgcnn_aflow_transfer_model(mat_prop, device):
    args = input_parser()
    args.device = device
    args.optim = 'AdamW'
    args.epochs = 250
    args.fea_path = "data/embeddings/matscholar-embedding.json"
    model_name = f'{mat_prop}_cgcnn_aflow_transfer_model_{args.epochs}_epochs'
    trained = f'models/best_{mat_prop}_cgcnn_aflow_model_{args.epochs}_epochs'
    csv_train = f'aflow/{mat_prop}/train.csv'
    csv_val = f'aflow/{mat_prop}/val.csv'
    csv_test = f'aflow/{mat_prop}/test.csv'
    # define dataset
    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)
    orig_atom_fea_len = dataset.atom_fea_dim
    args.fea_len = orig_atom_fea_len
    # train and test model
    train_roost(args, model_name, csv_train, csv_val=csv_val, transfer=trained)
    predict_roost(args, model_name, csv_test)


def show_results(model_name):
    df_results = pd.read_csv(f'results/test_results_{model_name}.csv')
    y_act, y_pred = df_results['target'], df_results['pred-0']
    # choose to "log10" or "unlog10" your data before metrics/plots
    if False:
        y_act, y_pred = np.log10(y_act), np.log10(y_pred)
    if False:
        y_act, y_pred = 10**y_act, 10**y_pred
    r2 = r2_score(y_act, y_pred)
    mae = mean_absolute_error(y_act, y_pred)
    print(f'-------------------')
    print(f'r2: {r2:0.4f}, mae: {mae:0.4f}')
    pva_plot(y_act, y_pred, model_name)


# %%

def run(i):
    if i == 0:
        # device = torch.cuda.device(1)
        device = 'cuda:0'
    else:
        device = 'cuda:1'
    mat_props = os.listdir('data/datasets/aflow')
    step = 4
    i = slice(i*step, step+i*step)
    for mat_prop in mat_props[i]:
        print('---------------------------------')
        print(f'evaluating property: {mat_prop}')
        print('---------------------------------')
        generate_standard_model(mat_prop, device)
        generate_cgcnn_aflow_trained_model(mat_prop, device)
        generate_cgcnn_aflow_transfer_model(mat_prop, device)


# if __name__ == '__main__':
#     device = 0
#     mat_props = os.listdir('data/datasets/aflow')
#     for mat_prop in mat_props[4:5]:
#         print('---------------------------------')
#         print(f'evaluating property: {mat_prop}')
#         print('---------------------------------')
#         generate_standard_model(mat_prop, device)
#         generate_cgcnn_aflow_trained_model(mat_prop, device)
#         generate_cgcnn_aflow_transfer_model(mat_prop, device)

