import os
import numpy as np
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
        start = time.time()
        train_loss, val_loss = [], []
        train_prec, train_rec, train_f1 = [], [], []
        val_prec, val_rec, val_f1 = [], [], []
        for bind, batch in enumerate(dl_train):
            print('Train: %d/%d  '%(bind+1, len(dl_train)), end='\r')
            detections, result_dict = model(batch[0], batch[2])
            train_loss.append(result_dict['loss'].item())
            opt.zero_grad()
            result_dict['loss'].backward()
            train_prec.append(result_dict['prec'])
            train_rec.append(result_dict['rec'])
            train_f1.append(result_dict['f1'])
            opt.step()
        print("")
        with torch.no_grad():
            for bind, batch in enumerate(dl_val):
                print('Val: %d/%d  '%(bind+1, len(dl_val)), end='\r')
                detections, result_dict = model(batch[0], batch[2])
                val_loss.append(result_dict['loss'].item())
                val_prec.append(result_dict['prec'])
                val_rec.append(result_dict['rec'])
                val_f1.append(result_dict['f1'])
        print("")

        avg_t_loss = sum(train_loss)/len(dl_train)
        avg_v_loss = sum(val_loss)/len(dl_val)
        avg_t_prec = np.array(train_prec)[:, 1].mean()
        avg_t_rec = np.array(train_rec)[:, 1].mean()
        avg_t_f1 = np.array(train_f1)[:, 1].mean()
        avg_v_prec = np.array(val_prec)[:, 1].mean()
        avg_v_rec = np.array(val_rec)[:, 1].mean()
        avg_v_f1 = np.array(val_f1)[:, 1].mean()

        end = time.time()
        etime = (end-start)/60
        msg1 = "Epoch: %d, Total Time: %.2fm\n"%(i+1, etime)
        msg2 = "Cross Entropy Loss ~ Train: %f, Val: %f\n"%(avg_t_loss, avg_v_loss)
        msg3 = "Precision ~ Train: %f, Val: %f\n"%(avg_t_prec, avg_v_prec)
        msg4 = "Recall ~ Train: %f, Val: %f\n"%(avg_t_rec, avg_v_rec)
        msg5 = "F1 ~ Train: %f, Val: %f\n"%(avg_t_f1, avg_v_f1)
        msg = msg1 + msg2 + msg3 + msg4 + msg5
        record(msg)
