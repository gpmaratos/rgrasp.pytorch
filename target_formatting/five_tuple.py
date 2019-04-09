import torch
import math
import torch.nn.functional as F

class rectangle:
    """
    Container which holds all important information about
    a rectangle, 32 comes from the size of the anchor box.
    """

    def __init__(self, x, y, w, l, t):
        anchor_size = 16
        self.x_pos = int(x // anchor_size)
        self.y_pos = int(y // anchor_size)
        #there seems to be an example rotating outside of the image so I clamp
        if self.x_pos == 20:
            self.x_pos = 19
        if self.y_pos == 20:
            self.y_pos = 19
        self.x_off = (x % anchor_size)/anchor_size
        self.y_off = (y % anchor_size)/anchor_size
        self.width = math.log(w/anchor_size)
        self.length = math.log(l/anchor_size)
        self.theta = math.log(t/3.145926)

class ConvertTuple:
    """
    Call this on a single rectangle to translate into a 5 tuple
    (x, y, w, l, t)
    """

    def __init__(self):
        pass

    def extract_vector(self, tup_a, tup_b):
        diff_x = tup_a[0] - tup_b[0]
        diff_y = tup_a[1] - tup_b[1]
        dist = math.sqrt(diff_x**2 + diff_y**2)
        if diff_x < 0:
            diff_y *= -1
            ang = math.acos(diff_y/dist)
        else:
            ang = math.acos(diff_y/dist)
        if ang == 0:
            import pdb;pdb.set_trace()
        return diff_x, diff_y, dist, ang

    def __call__(self, tup):
        x_cent = sum([coord[0] for coord in tup])/4
        y_cent = sum([coord[1] for coord in tup])/4
        x1, y1, d1, a1 = self.extract_vector(tup[0], tup[1])
        x2, y2, d2, a2 = self.extract_vector(tup[0], tup[-1])
        #let's put some rounding for precision reasons
#        x_cent = round(x_cent, 2)
#        y_cent = round(y_cent, 2)
#        d1 = round(d1, 2)
#        d2 = round(d2, 2)
        if d1 < d2:
#            a2 = round(a2, 2)
            return rectangle(x_cent, y_cent, d1, d2, a2)
        else:
#            a1 = round(a1, 2)
            return rectangle(x_cent, y_cent, d2, d1, a1)

class BalancedSampler:
    """
    Creates a balanced sample using the target rectangles for a batch.
    """
    def __init__(self, b_factor=2):
        self.b_factor = 2
        self.pred_off = []
        self.pred_cls = []
        self.targ_off = []
        self.targ_cls = []

    def clear_state(self):
        self.pred_off = []
        self.pred_cls = []
        self.targ_off = []
        self.targ_cls = []

    def extract(self, pred, target):
        """
        create balanced sample for a single pair
        To create the balanced pair, I need to choose a set of negative
        predictions. In this case I will sort the predictions and take
        the most confident false positives.

        there is a lot of room for error here. I am calling this function
        on all members of the batch, and appending extracted information
        into four different global lists
        """

        pos_size = len(target)
        neg_size = pos_size*self.b_factor

        #build target offsets, wierd formatting
        target_reg_t = [
            self.targ_off.append\
                ((rec.x_off, rec.y_off, rec.width, rec.length, rec.theta))\
                for rec in target
        ]

        #extract the positive samples from the targets
        prediction_reg = [
            self.pred_off.append(pred[:-2, rec.x_pos, rec.y_pos])\
                for rec in target
        ]
        prediction_cls_pos = [
            self.pred_cls.append(pred[-2:, rec.x_pos, rec.y_pos])\
                for rec in target
        ]

        #finish creating two class prediction
        #build negative cls samples
        pos_inds = [(rec.x_pos, rec.y_pos) for rec in target]
        cls_prediction = pred[-1, :, :]
        cls_prediction_view = cls_prediction.view(-1)
        cls_sort, cls_ind = torch.sort(cls_prediction_view, descending=True)
        i, count = 0, 0
        while count < neg_size:
            ind = cls_ind[i].item()
            pair = (int(ind // 20), int(ind % 20))
            if pair in pos_inds:
                i += 1
                continue
            self.pred_cls.append(pred[-2:, pair[0], pair[1]])
            count += 1
            i += 1
#        target_reg_t (target offsets), prediction_cls_pos/neg (target cls)

        [self.targ_cls.append(1) for i in range(pos_size)]
        [self.targ_cls.append(0) for i in range(neg_size)]

    def __call__(self, preds, targets):
        self.clear_state()
        balanced_sample = [self.extract(preds[i], targets[i])\
            for i in range(len(preds))]

        target_off = torch.tensor(self.targ_off)
        predicted_off = torch.stack(self.pred_off)
        target_cls = torch.tensor(self.targ_cls)
        predicted_cls = torch.stack(self.pred_cls)

        return target_off, target_cls, predicted_off, predicted_cls

class Loss:
    def __init__(self):
        pass

    def __call__(self, targ_off, targ_cls, pred_off, pred_cls):
        off = F.smooth_l1_loss(pred_off, targ_off)
        cls = F.cross_entropy(pred_cls, targ_cls)
        return off + cls, off, cls
