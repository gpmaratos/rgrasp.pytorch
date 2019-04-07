import math

class rectangle:
    """
    Container which holds all important information about
    a rectangle, 32 comes from the size of the anchor box.
    """

    def __init__(self, x, y, w, l, t):
        anchor_size = 16
        self.x_pos = int(x // anchor_size)
        self.y_pos = int(y // anchor_size)
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
        return diff_x, diff_y, dist, ang

    def __call__(self, tup):
        x_cent = sum([coord[0] for coord in tup])/4
        y_cent = sum([coord[1] for coord in tup])/4
        x1, y1, d1, a1 = self.extract_vector(tup[0], tup[1])
        x2, y2, d2, a2 = self.extract_vector(tup[0], tup[-1])
        if d1 < d2:
            return rectangle(x_cent, y_cent, d1, d2, a2)
        else:
            return rectangle(x_cent, y_cent, d2, d1, a1)

class BalancedSampler:
    """
    Creates a balanced sample using the target rectangles for a batch.
    """
    def __init__(self, b_factor=2):
        self.b_factor = 2

    def extract(self, pred, target):
        """create balanced sample for a single pair"""
        import pdb;pdb.set_trace()

    def __call__(self, preds, targets):
        balanced_sample = [self.extract(preds[i], targets[i])\
            for i in range(len(preds))]

