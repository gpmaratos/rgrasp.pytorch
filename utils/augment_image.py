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

    def translate_box(self, box, flip_horz, flip_vert, theta):
        ct = math.cos(theta)
        st = math.sin(theta)
        x = box[0] - self.x_dim/2
        y = box[1] - self.y_dim/2
        x = ct*x - st*y
        y = st*x + ct*y
        x += self.x_dim/2
        y += self.y_dim/2
        th = box[2]
        if flip_horz:
            x = self.x_dim - x
        if flip_vert:
            y = self.y_dim - y
        return (x, y, th)

    def __call__(self, n_image, b_boxes):
        """
        Updates an internal state (flips and rotations) before doing the
        rotations.
        """

        theta = round(random.uniform(self.min, self.max), 2)
        temp_image = n_image
        iarr = rotate(n_image, math.degrees(theta),
            preserve_range=True, order=0)
        theta *= -1
        if random.randint(0, 1):
            iarr = np.fliplr(iarr)
            flip_horz = True
        else:
            flip_horz = False
        if random.randint(0, 1):
            iarr = np.flipud(iarr)
            flip_vert = True
        else:
            flip_vert = False
        #b_boxes.ibboxes = [self.translate_box(box)\
        temp_boxes = [self.translate_box(box, flip_horz, flip_vert, theta)\
            for box in b_boxes.ibboxes]
        display_pair(temp_image, b_boxes, iarr, temp_boxes)
        return iarr, b_boxes
