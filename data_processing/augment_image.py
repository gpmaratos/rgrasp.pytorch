class Augmenter:
    def __call__(self, img, labels):
        return self.aug_fun(img, labels)

    def no_aug(self, img, labels):
        return img, labels

    def __init__(self, typ):
        aug_fun = {
            1:self.no_aug,
        }
        self.aug_fun = aug_fun[typ]
