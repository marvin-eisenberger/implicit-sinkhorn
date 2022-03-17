import torch
import numpy as np
import time
import scipy.io
import os.path

from utils.tools import *
from utils.img_tools import *
from image_barycenter.image_wasserstein import ImageWasserstein


class ImageBarycenter(ImageWasserstein):
    def __init__(self, args):
        super().__init__(args)

    def compute_barycenter(self, imgs_train, save_path=None):
        img = my_zeros(list(imgs_train.shape[-2:]))
        img = torch.nn.Parameter(img)

        loss_hist = []

        optimizer = torch.optim.Adam([img], lr=self.args.lr)
        np.set_printoptions(precision=3)

        t_s_ours = time.time()

        for it in range(self.args.num_epochs):
            optimizer.zero_grad()

            loss = 0
            for img_t in imgs_train:
                loss = loss + self.wasserstein_distance(torch.exp(img), img_t)
            loss.backward()

            optimizer.step()

            print("it {:d}/{:d}, loss = {:f} - Elapsed per it: {:f}".format(
                it+1, self.args.num_epochs, loss.item(), (time.time() - t_s_ours)/(it + 1)))

            loss_hist.append(loss.item())

            if it % self.args.log_freq == 0:
                if save_path is not None:
                    self.save_self(save_path, torch.exp(img).detach().cpu().numpy(), np.array(loss_hist), it)
                print_memory_status(it)

        return img

    def save_self(self, folder_path, img, loss_hist, it):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        mat_out = {'img': img,
                    'loss_hist': loss_hist,
                    'args': self.args.__dict__}

        mat_name = "data_" + str(it).zfill(6) + ".mat"
        mat_name = os.path.join(folder_path, mat_name)

        img_name = "img_" + str(it).zfill(6) + ".png"
        img_name = os.path.join(folder_path, img_name)

        loss_name = "loss_" + str(it).zfill(6) + ".png"
        loss_name = os.path.join(folder_path, loss_name)

        scipy.io.savemat(mat_name, mat_out)
        save_image(img_name, img)

        plt.plot(loss_hist)
        plt.savefig(loss_name)
        plt.close()

        print("Saved results to", folder_path)


def main_barycenter(args):
    img_arr = my_as_tensor([load_image(i, args.res) for i in args.image_indices])

    folder_out = save_path()

    bar = ImageBarycenter(args)
    bar.compute_barycenter(img_arr, folder_out)
