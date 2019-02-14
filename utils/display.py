import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def draw_tuple(draw, tup):
    #this is a copy from visualize
    print(tup)
    draw.ellipse((tup[0]-3, tup[1]-3, tup[0]+3, tup[1]+3),
        fill=(0, 0, 200))
    alpha = math.cos(tup[2])*10
    beta = math.sin(tup[2])*10
    x_ray = tup[0]+alpha
    y_ray = tup[1]+beta
    draw.line((tup[0], tup[1], x_ray, y_ray),
        fill=(0, 200, 0))

def display_pair(o_img, o_bbx, n_img, n_bbx):

        img = Image.fromarray(o_img)
        draw = ImageDraw.Draw(img)
        for i in range(len(o_bbx.irecs)):
            rec = o_bbx.irecs[i]
            tup = o_bbx.ibboxes[i]
            draw.line(rec[:2], fill=(200, 0, 0))
            draw.line(rec[1:], fill=(0, 0, 0))
            draw_tuple(draw, tup)

        n_img = n_img.astype(np.uint8)
        _img = Image.fromarray(n_img)
#        _draw = ImageDraw.Draw(_img)
#        [draw_tuple(_draw, tup) for tup in n_bbx]

        f, axarr = plt.subplots(2)
        axarr[0].imshow(img)
        axarr[1].imshow(_img)
        plt.show()
