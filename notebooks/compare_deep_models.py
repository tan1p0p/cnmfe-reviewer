# import packages
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from joblib import dump, load
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import optuna
import cnmfereview as cr

import config as cfg
from nn.model import Model
from nn.train import train, NNDataset


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name', type=str)
args = parser.parse_args()


ROOT = Path('../')
MODEL_LIST = [
    'OnlyCNN', 'LSTM', 'CNN_LSTM', # Only Spatial/Temporal
    'CNN', 'ResNet',               # Simple DNN
    'alexnet',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'squeezenet1_1',
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'mobilenet_v2',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2',
    'shufflenet_v2_x1_0',
    'googlenet',
]
MODE = 'fps'
if MODE == 'fps':
    DEVICE = 'cpu'
    PRETRAINED = False
else:
    DEVICE = 'cuda:0'
    PRETRAINED = True

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    seed=0
    fix_seed(seed)

    data = cr.Dataset(
        data_paths=cfg.data_paths,
        exp_id=cfg.exp_id,
        img_shape=cfg.img_shape,
        img_crop_size=cfg.img_crop_size,
        max_trace=cfg.max_trace_len,
    )

    x_train, x_test, y_train, y_test = data.split_training_test_data(
        test_split=.20, seed=10, for_deep=True)

    trainsets = NNDataset(x_train, y_train, DEVICE)
    testsets = NNDataset(x_test, y_test, DEVICE)
    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testsets, batch_size=32)

    for model_name in MODEL_LIST:
        print(f'\n======== {model_name} ========\n')
        if model_name == 'LSTM':
            model = Model(t_stage='LSTM', device=DEVICE, t_hidden_dim=500, t_output_dim=500, use_cnn_for_trace=False)
        elif model_name == 'CNN_LSTM':
            model = Model(t_stage='LSTM', device=DEVICE, t_hidden_dim=500, t_output_dim=500)
        elif model_name == 'OnlyCNN':
            model = Model(s_stage='CNN', device=DEVICE, block_num=5)
        else:
            model = Model(
                s_stage=model_name, t_stage='LSTM', device=DEVICE, pretrained=PRETRAINED,
                block_num=5, t_hidden_dim=500, t_output_dim=500
            )
        if MODE == 'train':
            score, model = train(model, model_name, train_loader, test_loader, DEVICE, log_path=f'{ROOT}/out/{model_name}.txt')
            model = model.to('cpu')
            torch.save(model.state_dict(), f'{ROOT}/best_models/{model_name}.pth')
        elif MODE == 'fps':
            model.eval()
            inputs = (torch.rand(1, 1, 500).to(DEVICE), torch.rand(1, 1, 80, 80).to(DEVICE))
            t0 = time.time()
            for i in range(100):
                model(inputs)
            with open(f'{ROOT}/out/speed.txt', 'a') as f:
                f.write(f'{model_name}: {100 / (time.time() - t0):.04f} fps\n')
        else:
            raise ValueError

if __name__ == '__main__':
    main()