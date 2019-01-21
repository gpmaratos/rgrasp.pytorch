import torch
from torch import optim
from data.CornellGrasp.dataloader import build_data_loader

def train(model, cfg, start=1, end=100):
    """
    Training loop for Grasp RCNN.
    """

    print('Beginning Training')
    if torch.cuda.is_available():
        dev = torch.device('cuda:1')
        print('Cuda detected, using GPU')
    else:
        dev = torch.dev('cpu')
        print('No cuda detected, using CPU')

    model = model.to(dev)
    dl = build_data_loader(cfg)
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr':1e-5},
        {'params': model.head.parameters()}
        ],lr=1e-4, momentum=0.9
    )

    for bind, batch in enumerate(dl):
        inp = batch[0].to(dev)
        model(inp, batch[1])
