"""@package docstring

These are all the methods and classes involved with translating the raw
data from the Cornell Grasp Dataset into objects that can be used for
training
"""

import os, random, cv2, math, re
import numpy as np
import argparse

class CornellTranslator:
    """
    Class that translates the raw images from the cornell
    dataset and serves data for either grasp detection or object detection.
    Requires the cornell data and z.txt from processedData.zip. Can
    serve images either object wise or image wise, and gives performs the
    train test split (which is always object wise)
    """

    def __init__(self, data_path, selected_set, split_type='image', seed=4211):
        """
        data_path (string): the path to the data (absolute or relative)

        selected_set (string): options are "train", "val", or "test".
            The parameter chooses which part of the dataset to serve.

        split_type (string): can be "image" or "object". Determines how
            the data is to be served.

        seed (int): chooses the random seed for spliting the data.
            4211 is the default and for my experiments I use this seed.
        """
        #set basic variables independent of task_type
        image_dimensions = (320, 320)
        augmenter = Augmenter(image_dimensions[0], image_dimensions[1])
        self.data_path = data_path
        self.image_dimensions = image_dimensions
        self.augmenter = augmenter
        #build object_lookup_table and background_lookup_table
        with open(os.path.join(data_path, 'z.txt')) as f:
            data = f.read().strip().split('\n')
        object_map, background_map = {}, {}
        for line in data:
            p_line = line.split()
            img_id, obj_num = int(p_line[0]), int(p_line[1])
            background_img = int(p_line[3].split('_')[1])
            if img_id >= 100 and img_id < 950 or\
                img_id >= 1000 and img_id < 1035:
                if not img_id in [132, 165]:
                    if obj_num in object_map:
                        if not img_id in object_map[obj_num]:
                            object_map[obj_num].append(img_id)
                            background_map[img_id] = background_img
                    else:
                        object_map[obj_num] = [img_id]
                        background_map[img_id] = background_img
        self.object_map = object_map
        self.background_map = background_map

        #CREATE THE NEW BACKGROUND MAP
        background_map = {}
        with open(os.path.join(data_path, 'backgrounds.txt')) as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                parsed_line = line.split('~')
                image_string = parsed_line[0].strip()
                background_string = parsed_line[1].strip()
                image_id = self.extract_id(image_string)
                background_id = self.extract_id(background_string)
                background_map[image_id] = background_id
        self.background_map = background_map

        #select dataset partition (all data splits are done using seed 4211, split is 70, 10, 20)
        random.seed(seed)
        obj_ids = list(object_map.keys())
        random.shuffle(obj_ids)
        split_factor = int(0.1*len(obj_ids))
        i = 7
        if selected_set == 'train':
            item_index = obj_ids[:i*split_factor]
        if selected_set == 'val':
            item_index = obj_ids[i*split_factor:(i+1)*split_factor]
        if selected_set == 'test':
            item_index = obj_ids[(i+1)*split_factor:]
        #create the index for either object wise or image wise split
        if split_type == 'image':
            item_index = [id for i in item_index for id in self.object_map[i]]
            self.is_image_type = True
        else:
            self.is_image_type = False
        self.item_index = np.array(item_index)

    def extract_id(self, name):
        """
        Simple helper function that takes the raw image name and gets the id.
        Assumes there is only one number embeded in the string.

        name (string): name of the image
        """
        image_id = re.findall('\d+', name)[0]
        image_id = int(image_id)
        return image_id

    def __len__(self):
        """Returns the number of total images that can be served"""
        return len(self.item_index)

    def __getitem__(self, index):
        """
        Overloaded square brackets, returns processed image[s].

        index (int): position in the dataset object
        """

        #if this class does object-wise splits then indexes are processed differently
        if not self.is_image_type:
            object_id = self.item_index[index]
            image_ids = self.object_map[object_id]
            processed_input = [self.extract_objects(image_id) for image_id in image_ids]
        else:
            image_id = self.item_index[index]
            processed_input = self.extract_image(image_id)
        return processed_input

    def extract_image(self, image_id):
        """
        image_id (int): which is specified in the name of the image pcd<id>, to
            extract it from the data folder
        """

        #read image
        path_prefix = os.path.join(self.data_path, 'pcd%04d'%(image_id))
        image_path = path_prefix + 'r.png'
        image_array = cv2.imread(image_path)
        #subtract background
        background_id = self.background_map[image_id]
        background_path = os.path.join(self.data_path, 'pcdb%04d'%(background_id)+'r.png')
        background_image = cv2.imread(background_path)
        image_array -= background_image
        #crop the image
        image_array = image_array[100:-60, 120:-200, :]
        #extract the ground truth
        with open(path_prefix + 'cpos.txt') as f:
            f = f.read().strip().split('\n')
        rectangles = [
            [self.get_coord(f[i]),
            self.get_coord(f[i+1]),
            self.get_coord(f[i+2]),
            self.get_coord(f[i+3])]
                for i in range(0, len(f), 4)
        ]
        #apply transformations (they will vary depending on the task) and return
        modified_array, modified_rectangles = self.augmenter(image_array, rectangles)
        return modified_array, modified_rectangles

    def get_coord(self, f):
        """
        Helper function that reads the coordinates of a rectangle from a string

        f (string): Rectangle Coordinates
        """

        #image cropping offsets are subtracted here
        ln = f.split()
        return (float(ln[0]) - 120, float(ln[1]) - 100)

    def yield_batches(self, size, randomize=True):
        """
        Yields batches of images and their labels.
        Also returns the percentage of the data completed

        Randomizer is currently unimplemented

        size (int): desired size of the batches
        """

        index = list(range(len(self)))
        if randomize:
            random.shuffle(index)
        i = 0
        while i < self.__len__():
            if (i + size - 1) < self.__len__():
                batch = [self[index[j]] for j in range(i, i+size)]
                percentage = (i+size)/self.__len__()
            else:
                batch = [self[index[j]] for j in range(i, self.__len__())]
                percentage = 1.
            images = [example[0] for example in batch]
            labels = [example[1] for example in batch]
            yield (images, labels, percentage)
            i += size


class Augmenter:
    """
    Defines an object that takes an image and its labels, and
    returns those images after a series of random augmentations.
    """

    def __call__(self, img, labels):
        """
        img (numpy): image to be augmented as a numpy array

        labels (list): list? of rectangles, rectangles are tuples of floats
        """

        img, labels = self.rotate(img, labels)
        img, labels = self.flip[random.randint(1, 4)](img, labels)
        #this is needed because pytorch doesn't support negative indices
        img = img.copy()
        return img, labels

    def translate_coord(self, coord, ct, st):
        """
        Helper function to project a coordinate, using a standard
        rotation matrix.

        coord (tup): tuple of floats representing a point to be translated

        ct, st (float): values calculated in the caller function that are used
        for the transformation cos(angle), sin(angle)
        """

        x = coord[0] - self.x_dim
        y = coord[1] - self.y_dim
        x = ct*x - st*y
        y = st*x + ct*y
        x += self.x_dim
        y += self.y_dim
        return (x, y)

    def rotate(self, img, labels):
        """
        Rotates an image and its labels. Randomly samples an angle between
        -30 and 30 degrees, rotates the image using opencv, and transforms
        the labels using a standard rotation matrix

        img (numpy): image to be rotated

        labels (list): labels to be rotated
        """
        #called in every function to rotate the image labels are assumed to be 4 points per rec
        theta = round(random.uniform(-30, 30), 2)
        rotation_matrix = cv2.getRotationMatrix2D((180, 180), theta, 1.0)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (320, 320))
        #multiply by -1
        theta = -1*math.radians(theta)
        ct = math.cos(theta)
        st = math.sin(theta)
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
        positive_labels = [ [self.translate_both(coord) for coord in label]
            for label in labels
        ]
        return img, labels

    def __init__(self, x, y):
        """
        Uses a mapping between a randomly sampled integer between 1 and 4
        inclusive to decide how to transform the object.

        x, y (int): height and width of the images to be translated.
            The augmenter assumes that all images are the same dimensions.
        """
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
