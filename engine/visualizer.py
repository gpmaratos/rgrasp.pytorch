import random
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset
from PIL import Image
from PIL import ImageDraw

def create_img(img, recs):
    img = Image.fromarray(img, 'RGB')
    draw = ImageDraw.Draw(img)
    for rec in recs:
        draw.line(rec, fill=(200, 0, 0))
    return img

def visualize(d_path):

    #set script variables
    ds = CornellDataset(d_path, 'train')
    #reset the random seed because dataset object sets it
    random.seed()
    for bind, batch in enumerate(ds):
        _, ax = plt.subplots(len(batch), 2)
        for i in range(len(batch)):
            img = create_img(batch[i][1], batch[i][3] )
            img_mod = create_img(batch[i][0], batch[i][2])
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(img_mod)
        plt.show()
