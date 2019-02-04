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
        ],lr=1e-2, momentum=0.9
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
    import pdb;pdb.set_trace()
