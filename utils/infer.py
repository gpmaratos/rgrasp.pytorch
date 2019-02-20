import math
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.transforms import Normalize

class Infer:
    """
    Infer. Class which runs the inference module of this project. It
    will take an image, use an existing detector to predict grasps, then
    display them as annotations on the original image. Formats and
    normalizes images in the style specified for pytorch pretrained
    models.
    """

    def __init__(self, cfg, model):

        #needs to be aware of number of anchors
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225]
        )
        chars = model.get_characteristics()
        pstride = chars['pstride']
        astride = chars['astride']
        num_ang = cfg.n_ang

        self.normalize = normalize
        self.num_ang = num_ang
        self.pstride = pstride
        self.astride = astride

    def draw_tuple(self, draw, tup):
        #this is a copy from visualize
        draw.ellipse((tup[0]-3, tup[1]-3, tup[0]+3, tup[1]+3),
            fill=(0, 0, 200))
        alpha = math.cos(tup[2])*10
        beta = math.sin(tup[2])*10
        x_ray = tup[0]+alpha
        y_ray = tup[1]+beta
        draw.line((tup[0], tup[1], x_ray, y_ray),
            fill=(0, 200, 0))

    def __call__(self, model, inp):
        """
        Infer. Runs inference using the trained model. Runs a very slow
        nested for loop to check every anchor location for a predicted
        grasp, then displays the annotated input
        """
        xtensor = torch.tensor(inp).permute(2, 0, 1).float()
        xtensor = self.normalize(xtensor)
        xtensor = xtensor.unsqueeze(0)
        pred = model(xtensor)
        pred = pred[0]

        preds = []
        for i in range(pred.shape[1]):
            for j in range(pred.shape[2]):
                for k in range(3, pred.shape[3], self.num_ang):
                    #change name of this variable so I do not exceed line
                    prediction = pred[0, i, j, k-3:k+1]
                    if prediction[3] > 0.9:
                        x = i*self.pstride + prediction[0].item()*self.pstride
                        y = j*self.pstride + prediction[1].item()*self.pstride
                        t = k/4*self.astride + prediction[2].item()*self.astride
                        preds.append((x, y, t))
        img = Image.fromarray(inp)
        draw = ImageDraw.Draw(img)
        [self.draw_tuple(draw, tup) for tup in preds]
        plt.imshow(img)
        plt.show()