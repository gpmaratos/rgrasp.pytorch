from math import radians, cos, sin
import numpy as np
from random import randint, uniform
from skimage.transform import rotate

class Augmenter:
    """
    Augmenter. takes the input and labels and outputs augmentations
    based on the integer it received
    """

    def __call__(self, img, labels):
        img, labels = self.rotate(img, labels)
        img, labels = self.flip[randint(1, 4)](img, labels)
        #this is needed because pytorch doesn't support negative indices
        img = img.copy()
        return img, labels

    def translate_coord(self, coord, ct, st):
        x = coord[0] - self.x_dim
        y = coord[1] - self.y_dim
        x = ct*x - st*y
        y = st*x + ct*y
        x += self.x_dim
        y += self.y_dim
        return (x, y)

    def rotate(self, img, labels):
    #called in every function to rotate the image labels are assumed to be 4 points per rec
        theta = round(uniform(self.min, self.max), 2)
        img = rotate(img, theta, preserve_range=True, order=0).astype(np.uint8)
        #multiply by -1 
        theta = -1*radians(theta)
        ct = cos(theta)
        st = sin(theta)
        labels = [[self.translate_coord(coord, ct, st) for coord in box]\
            for box in labels
        ]
        return img, labels

    def no_flip(self, img, labels):
        return img, labels

    #lots of duplicate code here, will fix this shamefulness later
    def translate_horz(self, coord):
        x_1 = self.x_dim*2 - coord[0]
        return (x_1, coord[1])

    def translate_vert(self, coord):
        y_1 = self.y_dim*2 - coord[1]
        return (coord[0], y_1)

    def translate_both(self, coord):
        x_1 = self.x_dim*2 - coord[0]
        y_1 = self.y_dim*2 - coord[1]
        return (x_1, y_1)

    def flip_horz_only(self, img, labels):
        img = np.fliplr(img)
        labels = [ [self.translate_horz(coord) for coord in label]
            for label in labels
        ]
        return img, labels

    def flip_vert_only(self, img, labels):
        img = np.flipud(img)
        labels = [ [self.translate_vert(coord) for coord in label]
            for label in labels
        ]
        return img, labels

    def flip_both(self, img, labels):
        img = np.fliplr(img)
        img = np.flipud(img)
        labels = [ [self.translate_both(coord) for coord in label]
            for label in labels
        ]
        return img, labels

    def __init__(self, x, y):
    #augment needs to know the dimensions of the input
        flip = {
            1:self.no_flip,
            2:self.flip_horz_only,
            3:self.flip_vert_only,
            4:self.flip_both,
        }
        self.flip = flip
        self.max = 30
        self.min = -30
        self.x_dim = x/2
        self.y_dim = y/2
