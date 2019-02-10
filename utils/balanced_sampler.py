import torch
from random import randint
from utils.gt_extractor import build_gt_extractor

def build_balanced_sampler(cfg):
    return BalancedSampler(cfg)

class BalancedSampler:
    """
    BalancedSampler. Class which takes network predictions and targets,
    and returns new predictions and targets which are tensors that are
    balanced (positive and negative) subsamples. The magic number 4
    comes from the fact that a prediction involves positions x, y, t
    and a classification score.

    Arguments:
        cfg
    """
    def __init__(self, cfg):
        super(BalancedSampler, self).__init__()
        gt_extractor = build_gt_extractor(cfg)
        dev = cfg.dev
        stride_factor = torch.tensor([gt_extractor.pixel_stride,
            gt_extractor.pixel_stride, gt_extractor.angle_stride]).to(dev)
        b_factor = cfg.b_factor
        self.gt_extractor = gt_extractor
        self.stride_factor = stride_factor
        self.pos_inds = []
        self.pos_examples = 0
        self.dev = dev
        self.b_factor = b_factor

    def clear_state(self):
        self.pos_inds = []
        self.pos_examples = 0

    def _extract_predictions(self, preds, targs):
        #builds pos_inds which is needed for extract negative
        inds = [(i, j, k) for i, j, k in targs[0]]
        self.pos_inds.append(inds)
        preds = [preds[i, j, 4*k:4*k+4] for i, j, k in inds]
        preds = torch.stack(preds)
        preds_reg = preds[:, :3]
        preds_cls = preds[:, 3]
        self.pos_examples += len(preds_cls)
        return preds_reg*self.stride_factor, preds_cls

    def extract_predictions(self, predictions, targets):
        #creates balanced samples of regression and cls
        preds = [self._extract_predictions(predictions[i], targets[i])\
                for i in range(len(targets))]
        preds_cls_neg = [
            self._extract_negative(predictions[i], self.pos_inds[i])
            for i in range(len(self.pos_inds))
        ]
        preds_reg = torch.cat([pred[0] for pred in preds])
        preds_cls = [pred[1] for pred in preds]
        preds_cls += preds_cls_neg
        preds_cls = torch.cat(preds_cls)
        return preds_reg, preds_cls

    def _extract_negative(self, preds, targs):
        #I will sample random values not in the positive examples
        #i suspect this is the slowest part of the algorithm
        inds = []
        while len(inds) < self.b_factor*len(targs):
            sample = (randint(0, preds.shape[0]-1),
                        randint(0, preds.shape[1]-1),
                        randint(0, (preds.shape[2]//4)-1))
            if sample not in targs:
                inds.append(sample)
        preds_cls_neg = [preds[i, j, 4*k+3] for i, j, k in inds]
        preds_cls_neg = torch.stack(preds_cls_neg)
        return preds_cls_neg

    def build_targets(self, targets):
        targ_reg = [torch.tensor(target[1]) for target in targets]
        targ_reg = torch.cat(targ_reg, dim=0)
        targ_cls = torch.zeros(self.pos_examples+\
            self.pos_examples*self.b_factor)
        targ_cls[:self.pos_examples] = 1.
        return targ_reg, targ_cls

    def __call__(self, predictions, targets):
        targets = [self.gt_extractor(target) for target in targets]
        self.clear_state()
        preds_reg, preds_cls = self.extract_predictions(
                                    predictions, targets)
        targ_reg, targ_cls = self.build_targets(targets)
        return preds_reg, preds_cls,\
            targ_reg.to(self.dev), targ_cls.to(self.dev),\
            self.pos_examples
