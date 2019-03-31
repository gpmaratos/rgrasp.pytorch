import os
import torch
import logging
import datetime
from torch import optim
from data_processing.dataloader import build_data_loader
from utils.cuda_check import get_device
from models.grasp_rcnn import build_model

def record(msg):
        print(msg)
        logging.info(msg)

def train(d_path, w_path, batch_size):
    """
    Training loop for grasping model. Model and optimizer are instantiated here.
    Results of each epoch are reported to stdout, and logged to 'train.log'.
    When training is done, the model is serialized to a file in the directory
    specified in w_path.
    """

    #set script variables
    device = get_device()
    epochs = 1
    dl_train = build_data_loader(batch_size, d_path, 'train')
    dl_val = build_data_loader(batch_size, d_path, 'val')
    logging.basicConfig(filename='train.log', level=logging.INFO)

    #create model and optimizer
    model = build_model(device)
    opt = optim.SGD([
        {'params': model.backbone.parameters(), 'lr':1e-5},
        {'params': model.head.parameters()}
        ],lr=1e-3, momentum=0.9, weight_decay=1e-4
    )

    #run training loop
    logging.info('')
    record('Training Session: %s'%(datetime.datetime.now()))
    for i in range(epochs):
        print('Epoch: %d'%(i+1))
        train_loss, val_loss = [], []
        for bind, batch in enumerate(dl_train):
            preds, loss = model(batch[0], batch[2])
            print(' %d/%d '%(bind, len(dl_train)), end='\r')
            train_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('')
        with torch.no_grad():
            for bind, batch in enumerate(dl_val):
                preds, loss = model(batch[0], batch[2])
                print(' %d/%d '%(bind, len(dl_val)), end='\r')
                val_loss.append(loss.item())
        print('')
        avg_train_loss = sum(train_loss)/len(dl_train)
        avg_val_loss = sum(val_loss)/len(dl_val)
        msg = ' Epoch: %d Train: %f Val: %f'%(i+1, avg_train_loss, avg_val_loss)
        record(msg)
    print('')

    model = model.to(torch.device('cpu'))
    m_pref = 'model-' + str(datetime.date.today())
    w_path = os.path.join(w_path, m_pref+'.pt')
    record("\nSaving Model at: %s"%(w_path))
    torch.save(model.state_dict(), w_path)
