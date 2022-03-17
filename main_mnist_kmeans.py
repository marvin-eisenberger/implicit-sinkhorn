import argparse

from image_barycenter.img_mnist_kmeans import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST k-means clustering.')

    parser.add_argument("--num_sink", type=int, default=10, help="number of Sinkhorn iterations")
    parser.add_argument("--lambd_sink", type=float, default=1e-3, help="the entropy regularization weight lambda")
    parser.add_argument("--num_epochs", type=int, default=100, help="the number of outer optimization steps/epochs")
    parser.add_argument("--lr", type=float, default=1e-1, help="the learning rate of the optimizer")
    parser.add_argument("--entropy_reg", type=bool, default=True, help="whether the entropy regularizer if used in the objective")
    parser.add_argument("--backward_type", type=str, default="impl", help="the gradient module used for optimization")
    parser.add_argument("--batch_size", type=int, default=1024, help="the batch size")
    parser.add_argument("--batch_subdivisions", type=int, default=4, help="subdivisions of each batch (to decrease memory demand)")
    parser.add_argument("--num_centroids", type=int, default=25, help="number of cluster centroids")

    args = parser.parse_args()
    main_kmeans_mnist(args)
