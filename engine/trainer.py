import os
import datetime
import torch
import pickle
from torch import optim
from data.CornellGrasp.dataloader import build_data_loader

def train(model, cfg, start=1, end=101):
    """
    Training loop for Grasp RCNN.
    """

    dev = cfg.dev
    model = model.to(dev)
    dl = build_data_loader(cfg)
    opt = optim.SGD([
        {'params': model.backbone.parameters(), 'lr':1e-5},
        {'params': model.head.parameters()}
        ],lr=1e-3, momentum=0.9, weight_decay=1e-4
    )

    for i in range(start, end):
        print("Epoch: %d"%(i))
        loss_history = []
        for bind, batch in enumerate(dl):
            inp = batch[0].to(dev)
            preds, loss = model(inp, batch[1])
            print(" %d/%d "%(bind, len(dl)), end='\r')
            loss_history.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(" Avg Loss: %f"%(sum(loss_history)/len(dl)))
    print('')
    m_pref = 'model-' + str(datetime.date.today())
    w_path = os.path.join(cfg.w_path, m_pref+'.pt')
    print("Saving Model at: %s"%(w_path))
    #maybe I should convert the model to cpu before serializing
    torch.save(model.state_dict(), w_path)
    with open(os.path.join(cfg.w_path, m_pref+'-cfg.pkl'), 'wb') as f:
        pickle.dump(cfg, f)
