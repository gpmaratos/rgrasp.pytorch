import os
import torch
import time
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
        ],lr=1e-4, momentum=0.9, weight_decay=1e-4
    )

    #run training loop
    logging.info('')
    log_msg = input("Message for log: ")
    record('Training Session: %s - %s'%(datetime.datetime.now(), log_msg))
    for i in range(epochs):
        print("")
        train_detect, val_detect = [], []
        start = time.time()
        train_loss, val_loss = [], []
        for bind, batch in enumerate(dl_train):
            print('Train: %d/%d  '%(bind+1, len(dl_train)), end='\r')
            out, loss = model(batch[0], batch[2])
            train_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("")
        with torch.no_grad():
            for bind, batch in enumerate(dl_val):
                print('Val: %d/%d  '%(bind+1, len(dl_val)), end='\r')
                out, loss = model(batch[0], batch[2])
                val_loss.append(loss)
        print("")
        avg_t_loss = sum(train_loss)/len(dl_train)
        avg_v_loss = sum(val_loss)/len(dl_val)
        end = time.time()
        etime = (end-start)/60
        msg = "Epoch: %d, Train: %f, Val: %f, Total Time: %.2fm"%(
            i+1, avg_t_loss, avg_v_loss, etime)
        record(msg)
