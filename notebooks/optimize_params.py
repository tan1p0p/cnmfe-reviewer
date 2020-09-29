import os
from pathlib import Path

from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

import cnmfereview as cr
import config as cfg
from nn.model import Model
from nn.train import NNDataset, train

DEVICE = 'cuda:0'

def search_res_block_num(train_loader, test_loader):
    print('\n=== ResNet ===')
    scores = []
    for i in range(1, 6):
        model = Model(
            s_stage='ResNet',
            res_block_num=i,
        )
        best_auc = train(model, train_loader, test_loader, DEVICE)
        scores.append(best_auc)

    for i in range(5):
        print(i+1, scores[i])

def search_san_params(train_loader, test_loader):
    print('\n=== SAN ===')
    def optimaze_san(trial):
        block_num = trial.suggest_int('block_num', 1, 5)
        layer_size_hop = trial.suggest_int('layer_size_hop', 2, 5)
        kernel_size = trial.suggest_int('kernel_size', 3, 7, 2)

        layers = [3]
        kernels = [3]
        for i in range(1, block_num):
            layers.append(2 + i*layer_size_hop)
            kernels.append(kernel_size)
        
        model = Model(
            s_stage='SAN',
            san_layers=layers,
            san_kernels=kernels,
        )
        score = train(model, train_loader, test_loader, DEVICE)
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(optimaze_san, n_trials=30)

def search_lstm_params(train_loader, test_loader):
    print('\n=== LSTM ===')
    def optimaze_lstm(trial):
        model = Model(
            s_stage='ResNet',
            res_block_num=4,
            t_hidden_dim=trial.suggest_int('t_hidden_dim', 50, 500, 50),
            t_output_dim=trial.suggest_int('t_output_dim', 50, 500, 50),
        )
        score = train(model, train_loader, test_loader, DEVICE)
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(optimaze_lstm, n_trials=30)

def main():

    data = cr.Dataset(
        data_paths=cfg.data_paths,
        exp_id=cfg.exp_id,
        img_shape=cfg.img_shape,
        img_crop_size=cfg.img_crop_size,
        max_trace=cfg.max_trace_len,
    )

    x_train, x_test, y_train, y_test = data.split_training_test_data(
        test_split=.20, seed=10, for_deep=True)
    print(f"Number of samples in training set: {len(x_train[0])}") 
    print(f"Number of samples in test set: {len(x_test[0])}")

    trainsets = NNDataset(x_train, y_train, DEVICE)
    testsets = NNDataset(x_test, y_test, DEVICE)
    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testsets, batch_size=32)

    # search_res_block_num(train_loader, test_loader) # 5 is best
    search_san_params(train_loader, test_loader)
    search_lstm_params(train_loader, test_loader)

if __name__ == "__main__":
    main()
