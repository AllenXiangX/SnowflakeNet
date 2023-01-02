# Point Cloud Upsampling

This repository contains the source code for the papers:

1. Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer (TPAMI 2022)

2. SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021, Oral)


## Datasets

We use the [PUGAN](https://github.com/liruihui/PU-GAN) dataset in our experiments, which are available below:

- [PUGAN: training h5 file](https://drive.google.com/file/d/13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ/view)
- [PUGAN: testing meshes](https://drive.google.com/file/d/1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC/view)

To generation the testing point clouds (.xyz files), please refer to the [PUGAN](https://github.com/liruihui/PU-GAN) repo.

## Getting Started

To use our code, make sure that the environment and PyTorch extensions are installed according to the instructions in the [main page](https://raw.githubusercontent.com/AllenXiangX/SnowflakeNet). Then modify the dataset path in the [configuration files](https://github.com/AllenXiangX/SnowflakeNet/tree/main/PU/configs).


## Training

To train a point cloud completion model from scratch, run:

```
export CUDA_VISIBLE_DEVICES='0'

python train.py
```


## Evaluation

To evaluate a pre-trained model, first specify the model_path in configuration file, then run:

```
export CUDA_VISIBLE_DEVICES='0'

python test.py
```


## Acknowledgements


This repo is based on: 
- [PUGAN](https://github.com/liruihui/PU-GAN)

We thank the authors for their great job!