import os
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import numpy as np
import cv2

from utils.tools import *


def load_image(img_id=0, resolution=32):
    data_folder = os.path.join("image_barycenter", "data")

    if img_id == 0:
        img_file = 'circle.png'
    if img_id == 1:
        img_file = 'star.png'
    if img_id == 2:
        img_file = 'arrow.png'
    if img_id == 3:
        img_file = 'torus.png'
    if img_id == 4:
        img_file = 'circle_corner.png'
    if img_id == 5:
        img_file = 'two_circles.png'

    img = pltimg.imread(os.path.join(data_folder, img_file))

    img = img[:, :, 0]
    img = rescale_img(1-img)
    img = cv2.resize(img, dsize=(resolution, resolution), interpolation=cv2.INTER_NEAREST)
    return img


def save_image(img_file, img):
    plt.imshow(img)
    plt.savefig(img_file)
    plt.close()


def show_image(img):
    _ = plt.imshow(img.detach().cpu().numpy())
    plt.show()


def rescale_img(img, eps=1e-5):
    return img / (1-2*eps) + eps


def img_gradient_batch_centraldiff(img):
    g_row = img[:, 2:, :] - img[:, :-2, :]
    g_col = img[:, :, 2:] - img[:, :, :-2]
    return g_row, g_col
