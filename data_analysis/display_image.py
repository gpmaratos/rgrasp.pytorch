"""
Script to display a single image with its image id
"""


import numpy as np
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds_train = CornellDataset(d_path, 'train')
    ds_val = CornellDataset(d_path, 'val')
    ds_test = CornellDataset(d_path, 'test')

    print("View Object Batches with their labels. Enter 'continuous' to view",
        " all objects sequentially")
    print("--enter 'exit' to exit")
    print("--enter 'train' 'val' or 'test' to switch dataset to view")

    dataset = 'train'
    ds = ds_train

    while True:
        print("\nCurrent Dataset <%s>"%(dataset))
        inp = input("Select your input, range (0 - %d): "%(len(ds)))
        if inp == 'exit':
            return
        if inp == 'train':
            dataset = 'train'
            ds = ds_train
            continue
        if inp == 'val':
            dataset = 'val'
            ds = ds_val
            continue
        if inp == 'test':
            dataset = 'test'
            ds = ds_test
            continue
        if inp == 'continuous':
            for bind, batch in enumerate(ds):
                _, ax = plt.subplots(len(batch)+1, 2)
                for i in range(len(batch)):
                    img = create_img(batch[i][1], batch[i][3] )
                    img_mod = create_img(batch[i][5], batch[i][4])
                    ax[i, 0].imshow(img)
                    ax[i, 1].imshow(img_mod)
                print("Displaying image: %d"%(bind))
                plt.show()
            continue
        try:
            identifier = int(inp)
            batch = ds[identifier]
            _, ax = plt.subplots(len(batch)+1, 2)
            for i in range(len(batch)):
                img = create_img(batch[i][1], batch[i][3] )
                img_mod = create_img(batch[i][5], batch[i][4])
                ax[i, 0].imshow(img)
                ax[i, 1].imshow(img_mod)
            plt.show()

        except ValueError:
            print("Error: %s is not a valid id or command!"%(inp))
            continue
main()
