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

def print_metrics(ds, loss, prec, rec, f1):
    avg_loss = sum(loss)/len(loss)
    avg_prec = sum(prec)/len(prec)
    avg_rec = sum(rec)/len(rec)
    avg_f1 = sum(f1)/len(f1)

    msg1 = 'Results for <%s>\n'%(ds)
    msg2 = 'Cross Entropy Loss ~ %f\n'%(avg_loss)
    msg3 = 'Precision and Recall ~ %f - %f\n'%(avg_prec, avg_rec)
    msg4 = 'F1 Score ~ %f\n'%(avg_f1)
    record(msg1+msg2+msg3+msg4)

def train(d_path, w_path):
    """
    Trains Cascade net which has two components, the detector and the
    predictor
    """

    #set script variables
    epochs = 500
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

        end = time.time()
        etime = (end-start)/60
        record('Epoch: %d, Total Time: %.2fm\n'%(i+1, etime))
        print_metrics('train', train_loss, train_prec, train_rec, train_f1)
        print_metrics('val', val_loss, val_prec, val_rec, val_f1)

    model = model.to(torch.device('cpu'))
    m_pref = 'model-' + str(datetime.date.today())
    w_path = os.path.join(w_path, m_pref+'.pt')
    record("\nSaving Model at: %s"%(w_path))
    torch.save(model.state_dict(), w_path)
