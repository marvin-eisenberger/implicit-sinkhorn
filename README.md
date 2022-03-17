<h1> A Unified Framework for Implicit Sinkhorn Differentiation </h1>

Implementation of the CVPR 2022 paper. We analyze the use of implicit gradients for generic Sinkhorn layers within a neural network. In the paper, we provide extensive theoretical and empirical analysis. Here, we provide the PyTorch source code for many of the experiments from the paper.

## Usage

### Sinkhorn module

* If you want to use our PyTorch Sinkhorn module (with implicit gradients) in your own project, simply include the file `sinkhorn/sinkhorn.py`.
* The module has minimal requirements, it only depends on PyTorch (tested for version 1.6.0).
```bash
pip3 install torch
```

### Setting up the repo
* For reproducing experiments from our paper, you have to install the whole repo.
* Our code mainly depends on fairly standard packages for Vision and DL, all of which can be installed with pip:
```bash
pip3 install torch torchvision numpy scipy opencv-python matplotlib
```
Alternatively, the repo can be installed directly via anaconda:
```bash
conda env create --name implicit_sinkhorn_env -f implicit_sinkhorn_env.yml
conda activate implicit_sinkhorn_env
```

## Experiments
We provide an implementation of two experimental settings, where the Sinkhorn module in `sinkhorn/sinkhorn.py` is at the core of both of them.
### Image barycenter
* To reproduce our experiments on barycentric interpolation of images, run
```bash
python3 main_image_barycenter.py 1 4
```
* The numbers '1' and '4' refer to the images `star.png` and `circle_corner.png` under `image_barycenter/data/`, respectively.
* Further possible ids are '0': `circle.png`, '2': `arrow.png`, '3': `torus.png`, '5': `two_circles.png`
* The script `main_image_barycenter.py` can also accept 3 or more image ids.
* The outputs will be saved automatically under `results/` in a new folder, associated with the timestamp of the script's execution. Every 1000 iterations, it saved a thumbnail of the output images, loss curves and some metadata (and raw image data) in `.mat` files.
* The 'Automatic Differentiation' baseline can be executed by passing a corresponding option:
```bash
python3 main_image_barycenter.py 1 4 --backward_type ad
```
* Additional options and parameters (such as number of Sinkhorn iterations, image resolution) are specified in `main_image_barycenter.py`. 


### MNIST k-means clustering
* This experiment requires the MNIST dataset. In the first run, it will be downloaded and stored  under `./image_barycenter/data`, which takes ~117MB of storage space. Alternatively, the dataset path can be specified at the end of the file `./image_barycenter/img_mnist_kmeans.py`, if it already exists on the current machine.
* To run the experiment, execute:
```bash
python3 main_mnist_kmeans.py
```
* The script requires no additional inputs. However, different options can be specified, equivalently to the image barycenter experiment.

### Citation
If you use our implementation, please cite:
```
@article{eisenberger2022unified,
  title={A Unified Framework for Implicit Sinkhorn Differentiation},
  author={Eisenberger, Marvin and Toker, Aysim and Leal-Taix{\'e}, Laura and Bernard, Florian and Cremers, Daniel},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
