import os
import torch
import logging
import datetime
from torch import optim
from data_processing.dataloader import build_data_loader
from utils.cuda_check import get_device
from models.cascade_net import build_model

def record(msg):
        print(msg)
        logging.info(msg)

def train(d_path, w_path):
    """
    Trains Cascade net which has two components, the detector and the
    predictor
    """

    #set script variables
    epochs = 100
    batch_size = 3
    device = get_device()
    #these still require number of angles for some reason
    dl_train = build_data_loader(batch_size, d_path, 'train', 3)
    dl_val = build_data_loader(batch_size, d_path, 'val', 3)
    logging.basicConfig(filename='train.log', level=logging.INFO)

    #create model and optimizer
    model = build_model(device)
    opt = optim.SGD([
        {'params': model.detector.parameters()}
        ],lr=1e-3, momentum=0.9, weight_decay=1e-4
    )

    #run training loop
    logging.info('')
    record('Training Session: %s'%(datetime.datetime.now()))
    for i in range(epochs):
        print('Epoch: %d'%(i+1))
        train_detect, val_detect = [], []
        for bind, batch in enumerate(dl_train):
            print('Train: %d/%d  '%(bind+1, len(dl_train)), end='\r')
            out = model(batch[0], batch[2])
