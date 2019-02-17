import math
import random
import numpy as np
from skimage.transform import rotate
from utils.display import display_pair

def build_augmenter(x_dim, y_dim):
    return Augment(x_dim, y_dim)

class Augment:
    """
    Augment. Class that does all the image augmentations that randomize
    the images for training.
    """
    def __init__(self, x_dim, y_dim):
        a_range = math.radians(30)
        self.max = a_range
        self.min = -1*a_range
        self.x_dim = x_dim
        self.y_dim = y_dim

    def translate_box(self, box):
        if self.flip_horz:
            x = box[0] - self.x_dim
        else:
            x = box[0]
        if self.flip_vert:
            y = box[1] - self.y_dim
        else:
            y = box[1]
        th = box[2] - self.theta
        return (x, y, th)

    def __call__(self, n_image, b_boxes):
        """
        Updates an internal state (flips and rotations) before doing the
        rotations.
        """

        self.theta = round(random.uniform(self.min, self.max), 2)
        temp_image = n_image
        iarr = rotate(n_image, self.theta, preserve_range=True, order=0)
        if random.randint(0, 1):
            iarr = np.fliplr(iarr)
            self.flip_horz = True
        else:
            self.flip_horz = False
        if random.randint(0, 1):
            iarr = np.flipud(iarr)
            self.flip_vert = True
        else:
            self.flip_vert = False
        #b_boxes.ibboxes = [self.translate_box(box)\
        temp_boxes = [self.translate_box(box)\
            for box in b_boxes.ibboxes]
        display_pair(temp_image, b_boxes, iarr, temp_boxes)
        return iarr, b_boxes