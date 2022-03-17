import argparse

from image_barycenter.img_barycenter import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Barycenter.')

    parser.add_argument('image_indices', type=int, nargs='+', help='the list of test image ids between [0-5]')

    parser.add_argument("--res", type=int, default=64, help="the resolution of the input images")
    parser.add_argument("--num_sink", type=int, default=100, help="number of Sinkhorn iterations")
    parser.add_argument("--lambd_sink", type=float, default=2e-3, help="the entropy regularization weight lambda")
    parser.add_argument("--num_epochs", type=int, default=20_001, help="the number of outer optimization steps/epochs")
    parser.add_argument("--lr", type=float, default=1e-1, help="the learning rate of the optimizer")
    parser.add_argument("--entropy_reg", type=bool, default=True, help="whether the entropy regularizer if used in the objective")
    parser.add_argument("--backward_type", type=str, default="impl", help="the gradient module used for optimization")
    parser.add_argument("--log_freq", type=int, default=1000, help="log frequency/number of epochs after which outputs are generated")

    args = parser.parse_args()
    main_barycenter(args)
