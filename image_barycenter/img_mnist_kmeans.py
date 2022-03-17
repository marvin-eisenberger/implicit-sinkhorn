import torch
import numpy as np
import time
import scipy.io
import os.path
import torchvision.datasets
import torchvision.transforms as transforms

from utils.img_tools import *
from utils.tools import *
from image_barycenter.image_wasserstein import ImageWasserstein


class ImageDatasetKmeans(ImageWasserstein):
    def __init__(self, args):
        super().__init__(args)

    def compute_kmeans(self, dataset, save_path=None):
        imgs = self.init_centroids(dataset)
        imgs = torch.nn.Parameter(imgs)

        loss_hist = []

        optimizer = torch.optim.Adam([imgs], lr=self.args.lr)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        np.set_printoptions(precision=3)

        for i_epoch in range(self.args.num_epochs):
            t_s_ours = time.time()

            for it, (img_arr, _) in enumerate(train_loader):
                img_arr = rescale_img(img_arr[:, 0, :, :])
                optimizer.zero_grad()

                tot_loss = 0

                n_sub = img_arr.shape[0] // self.args.batch_subdivisions

                with torch.no_grad():
                    dist_to_centroid = my_zeros((img_arr.shape[0], imgs.shape[0]))
                    for i_sub in range(self.args.batch_subdivisions):
                        i_from = i_sub * n_sub
                        i_to = (i_sub + 1) * n_sub

                        for i_centroid in range(imgs.shape[0]):
                            img_centroid = torch.exp(imgs[i_centroid, :, :].unsqueeze(0))
                            pi, d = self.sinkhorn_forward_pass(img_centroid, img_arr[i_from:i_to, :, :].to(device))
                            dist_to_centroid[i_from:i_to, i_centroid] = (d*pi).sum(dim=[1, 2])
                    assignment = dist_to_centroid.argmin(dim=1)

                for i_sub in range(self.args.batch_subdivisions):
                    i_from = i_sub * n_sub
                    i_to = (i_sub+1) * n_sub
                    img = torch.exp(imgs[assignment[i_from:i_to]])
                    loss = self.wasserstein_distance(img, img_arr[i_from:i_to, :, :].to(device))
                    loss = loss / self.args.batch_subdivisions
                    loss.backward()

                    tot_loss += loss.item()

                optimizer.step()

                print("epoch {:d}, it {:d}, loss = {:f} - Elapsed per it: {:f}".format(
                    i_epoch, it + 1, loss.item(), (time.time() - t_s_ours)/(it + 1)))

                loss_hist.append(tot_loss)

                if it == 0:
                    if save_path is not None:
                        self.save_self(save_path, torch.exp(imgs).detach().cpu().numpy(), np.array(loss_hist), it, i_epoch)
                    print_memory_status(it)

    def init_centroids(self, dataset, eps=1e-2):
        # we want to select the same centroids for each settings to make them more comparable
        # we skip 3 indexes to increase the diversity of initializations
        i_skip = [8, 14, 23]
        _, n_rows, n_cols = np.array(dataset[0][0]).shape
        imgs = my_zeros([self.args.num_centroids, n_rows, n_cols])

        i_dataset = 0
        i_centroid = 0

        while i_centroid < self.args.num_centroids:
            if i_dataset not in i_skip:
                imgs[i_centroid, :, :] = torch.log(eps + dataset[i_dataset][0][0, :, :])
                i_centroid += 1
            i_dataset += 1

        return imgs

    def save_self(self, folder_path, imgs, loss_hist, it, i_epoch):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        mat_out = {'imgs': imgs,
                'loss_hist': loss_hist,
                'args': self.args.__dict__}

        mat_name = "data_" + str(i_epoch).zfill(3) + "_" + str(it).zfill(5) + ".mat"
        mat_name = os.path.join(folder_path, mat_name)

        loss_name = "loss_" + str(i_epoch).zfill(3) + "_" + str(it).zfill(5) + ".png"
        loss_name = os.path.join(folder_path, loss_name)

        scipy.io.savemat(mat_name, mat_out)
        for i, img in enumerate(imgs):
            img_name = "img_" + str(i_epoch).zfill(3) + "_" + str(it).zfill(5) + "_" + str(i) + ".png"
            img_name = os.path.join(folder_path, img_name)
            save_image(img_name, img)

        plt.plot(loss_hist)
        plt.savefig(loss_name)
        plt.close()

        print("Saved results to", folder_path)


def main_kmeans_mnist(args):
    dataset = torchvision.datasets.MNIST("./image_barycenter/data", transform=transforms.ToTensor(), download=True)

    bar = ImageDatasetKmeans(args)
    bar.compute_kmeans(dataset, save_path())