# MobileODE: An Extra Lightweight Network

## Major component

Our COS module can be found in ```./modules/mobilenetv2.py```.

## Installation

Use Python 3.10+ and [PyTorch](https://pytorch.org) (version >= v1.12.0).

```bash
# Clone the original repo
git clone git@github.com:apple/ml-cvnets.git
cd ml-cvnets

# Create a virtual env. We use Conda
conda create -n cvnets python=3.10.8
conda activate cvnets

# install requirements and CVNets package
pip install -r requirements.txt -c constraints.txt
pip install --editable .
```

## Getting started
   * Add our file to the same path in [Original repo](https://github.com/apple/ml-cvnets)
   * Use the training commamd as follows.
```bash
# For classification (e.g.MobileODEV2)
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/classification/imagenet_tiny/mobilenetv2_ode.yaml --common.results-loc Results/imagenet_tiny/mobilenetv2_ode | tee -a Results/imagenet_tiny/mobilenetv2_ode.log


# For detection
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/BUSI_coco/mobilenetv2_ode.yaml --common.results-loc Results/BUSI_coco/mobilenetv2_ode --ddp.dist-port 29550  --common.override-kwargs | tee -a ./Results/imagenet_r/classification_mobilenetv4.log


# For segmentation
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/segmentation/ade20k/mobilenetv2_ode.yaml --common.results-loc Results/ade20k/mobilenetv2_ode | tee -a Results/ade20k/mobilenetv2_ode.log

```
