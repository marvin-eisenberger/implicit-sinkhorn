import torch

from utils.tools import save_path
from sinkhorn.sinkhorn import Sinkhorn, sinkhorn_unrolled
from utils.img_tools import *


class ImageWasserstein:
    def __init__(self, args):
        self.args = args

    def sinkhorn(self):
        if self.args.backward_type == "impl":
            return Sinkhorn.apply
        elif self.args.backward_type == "ad":
            return sinkhorn_unrolled

    def _img_grid(self, imgs):
        n, m = imgs.shape[-2:]
        grid_n, grid_m = torch.meshgrid(my_linspace(0, 1, n), my_linspace(0, 1, m))

        grid_n = torch.reshape(grid_n, [n * m, 1])
        grid_m = torch.reshape(grid_m, [n * m, 1])

        return torch.cat((grid_n, grid_m), dim=1)

    def _marginal(self, imgs):
        n, m = imgs.shape[-2:]
        b = list(imgs.shape[:-2])
        nu = torch.reshape(imgs, b + [n * m])
        nu = nu / (nu.sum(dim=-1, keepdim=True) * 1.)
        return nu

    def sinkhorn_forward_pass(self, img, imgs):
        b = imgs.shape[:-2]

        grid = self._img_grid(img)

        d = dist_mat(grid, grid)
        d = torch.reshape(d, [1]*len(b) + list(d.shape))
        d = d.repeat(b + (1, 1))

        nu = self._marginal(img)
        mu = self._marginal(imgs)

        pi = self.sinkhorn()(d, nu, mu, self.args.num_sink, self.args.lambd_sink)

        return pi, d

    def wasserstein_distance(self, img, imgs):
        pi, d = self.sinkhorn_forward_pass(img, imgs)

        dist_wass = (d*pi).sum(dim=(-2, -1)).mean()

        if self.args.entropy_reg:
            eps = 1e-5
            pi = pi + eps
            dist_wass = dist_wass + self.args.lambd_sink * (pi * (torch.log(pi)-1)).sum(dim=(-2, -1)).mean()

        return dist_wass
