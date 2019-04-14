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
    epochs = 1000
    dl_train = build_data_loader(batch_size, d_path, 'train', 3)
    dl_val = build_data_loader(batch_size, d_path, 'val', 3)
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
    for i in range(200):
        print('Epoch: %d'%(i+1))
        train_off, train_cls, val_off, val_cls = [], [], [], []
        train_offs, val_offs = torch.zeros(5), torch.zeros(5)
        for bind, batch in enumerate(dl_train):
            preds, loss, off, cls, offs = model(batch[0], batch[2])
            print(' %d/%d '%(bind, len(dl_train)), end='\r')
            train_off.append(off.item())
            train_cls.append(cls.item())
            train_offs += offs
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(' %d/%d '%(len(dl_train), len(dl_train)))
        with torch.no_grad():
            for bind, batch in enumerate(dl_val):
                preds, loss , off, cls, offs = model(batch[0], batch[2])
                print(' %d/%d '%(bind, len(dl_val)), end='\r')
                val_off.append(off.item())
                val_cls.append(cls.item())
                val_offs += offs
        print(' %d/%d '%(len(dl_val), len(dl_val)))
        avg_t_off = sum(train_off)/len(dl_train)
        avg_t_cls = sum(train_cls)/len(dl_train)
        avg_v_off = sum(val_off)/len(dl_val)
        avg_v_cls = sum(val_cls)/len(dl_val)
        train_offs /= len(dl_train)
        val_offs /= len(dl_val)
        msg = '\nEpoch: %d\n Train: [%s, %f]\n Val: [%s, %f] '%(i+1, train_offs,
            avg_t_cls, val_offs, avg_v_cls)
        record(msg)
    print('')

    #create model and optimizer
    opt = optim.SGD([
        {'params': model.backbone.parameters(), 'lr':1e-5},
        {'params': model.head.parameters()}
        ],lr=1e-5, momentum=0.9, weight_decay=1e-4
    )

    #run training loop
    logging.info('')
    record('Training Session: %s'%(datetime.datetime.now()))
    for i in range(200, 800):
        print('Epoch: %d'%(i+1))
        train_off, train_cls, val_off, val_cls = [], [], [], []
        train_offs, val_offs = torch.zeros(5), torch.zeros(5)
        for bind, batch in enumerate(dl_train):
            preds, loss, off, cls, offs = model(batch[0], batch[2])
            print(' %d/%d '%(bind, len(dl_train)), end='\r')
            train_off.append(off.item())
            train_cls.append(cls.item())
            train_offs += offs
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(' %d/%d '%(len(dl_train), len(dl_train)))
        with torch.no_grad():
            for bind, batch in enumerate(dl_val):
                preds, loss , off, cls, offs = model(batch[0], batch[2])
                print(' %d/%d '%(bind, len(dl_val)), end='\r')
                val_off.append(off.item())
                val_cls.append(cls.item())
                val_offs += offs
        print(' %d/%d '%(len(dl_val), len(dl_val)))
        avg_t_off = sum(train_off)/len(dl_train)
        avg_t_cls = sum(train_cls)/len(dl_train)
        avg_v_off = sum(val_off)/len(dl_val)
        avg_v_cls = sum(val_cls)/len(dl_val)
        train_offs /= len(dl_train)
        val_offs /= len(dl_val)
        msg = '\nEpoch: %d\n Train: [%s, %f]\n Val: [%s, %f] '%(i+1, train_offs,
            avg_t_cls, val_offs, avg_v_cls)
        record(msg)
    print('')

    model = model.to(torch.device('cpu'))
    m_pref = 'model-' + str(datetime.date.today())
    w_path = os.path.join(w_path, m_pref+'.pt')
    record("\nSaving Model at: %s"%(w_path))
    torch.save(model.state_dict(), w_path)
