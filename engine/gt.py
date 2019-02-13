from data.CornellGrasp.dataset import build_dataset
def gt(cfg):
    ds = build_dataset(cfg, train=False, aug=True)
    for example in ds:
        import pdb;pbd.set_trace()
