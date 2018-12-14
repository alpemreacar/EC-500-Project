# Deep Learning Course Project, Fall 2018

This repository contains [CycleGAN](https://junyanz.github.io/CycleGAN/) implementation. The repository has GAN and CycleGAN architectures and their test results.

The repository doesn't contain cycleGAN models and datasets due to their sizes. It has a simple GAN model in models folder.

Open jupyter file *code/v3_GAN.ipynb* to see GAN structure tested on MNIST. This model is saved, it can be used without training as shown in the jupyter notebook.

In addition to GAN and CycleGAN scripts, the repository also has processing code for pix2pix datasets, test results for a pretrained pix2pix model on day2nigth dataset and evaluation code for calculating MSE and SSIM scores.

Test results from datasets can be found in *code/v5_cycleGAN/Pictures/Test/*

### How to run?

  - Clone the repository
  - Put your dataset under foldder *data/dataset_name*. 
    - It should have subfolders as *testA*, *testB*, *trainA*, *trainB* with corresponding images.
  - Create folders named as:
    - *code/v5_cycleGAN/Pictures/Test/dataset_name/testA/*
    - *code/v5_cycleGAN/Pictures/Test/dataset_name/testB/*
    - *code/v5_cycleGAN/Pictures/Traning/dataset_name/*
    - *models/v5_CycleGAN/dataset_nam/e*
  - Modify dataset variable in *code/v5_cycleGAN/train.py* and *code/v5_cycleGAN/test.py* to *dataset_name*
  - Run *code/v5_cycleGAN/train.py*
    - It saves losses in *code/v5_cycleGAN/loss_arr/*
    - It saves random generated images after each epoch in *code/v5_cycleGAN/Pictures/Traning/dataset_name/*
    - It saves model in *models/v5_CycleGAN/dataset_name*
 - Run *code/v5_cycleGAN/test.py*
    - It saves generated images in *code/v5_cycleGAN/Pictures/Test/dataset_name/*
 - Run *code/v5_cycleGAN/plot.py* for a training loss plot
 
