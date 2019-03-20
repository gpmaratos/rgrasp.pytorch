import math

class GTExtractor:
    """
    GTExtractor. Class which reads the targets object, which is a batch
    of bbox objects, and returns two objects. The first is a batch of
    anchor positions, which are calculated for each grasp point. The
    second is the offsets from the anchor positions in the first batch,
    which is what the network learns to predict during training.
    It currently assumes that the backbone is a
    Resnet50. The backbone determines how many anchor boxes the RCNN uses.
    That is where the magic number 32 comes from.

    Arguments:
        n_ang (int): number of angles represented in the anchors


    """
    def __init__(self, n_ang):
        super(GTExtractor, self).__init__()
        pixel_stride = 32
        angle_stride = math.radians(180)/n_ang
        self.pixel_stride = pixel_stride
        self.angle_stride = angle_stride
        self.anchor_points = []
        self.offsets = []

    def extract_anchor(self, bbox):
        x = bbox[0] / self.pixel_stride
        y = bbox[1] / self.pixel_stride
        #this is required because rounding later can cause matching problems
        ang = round(bbox[2], 2)
        t = ang / self.angle_stride
        self.anchor_points.append((int(x), int(y), int(t)))
        x_off = bbox[0] % self.pixel_stride
        y_off = bbox[1] % self.pixel_stride
        t_off = bbox[2] % self.angle_stride
        self.offsets.append((x_off, y_off, t_off))

    def clear_points(self):
        self.anchor_points = []
        self.offsets = []

    def __call__(self, bboxes):
        self.clear_points()
        [self.extract_anchor(bbox) for bbox in bboxes.ibboxes]
        return self.anchor_points, self.offsets
