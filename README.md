# Deep Learning Course Project, Fall 2018

This repository contains [CycleGAN](https://junyanz.github.io/CycleGAN/) implementation code. The repository has GAN and CycleGAN architectures along with their test results.

The repository doesn't contain CycleGAN models and tested datasets due to their sizes. It has a simple GAN model in the *models* folder.

Open the jupyter file *code/v3_GAN.ipynb* to see the GAN structure tested on the MNIST dataset. This model is saved and it can be used without training as shown in the jupyter notebook.

In addition to GAN and CycleGAN scripts, the repository also has processing code for the day2night dataset, test results for a pretrained pix2pix model on the day2night dataset and the evaluation code for calculating MSE and SSIM scores.

Test results from the datasets can be found in *code/v5_cycleGAN/Pictures/Test/*.

### How to run?

  - Clone the repository.
  - Put your dataset under folder *data/dataset_name*. 
    - It should have subfolders as *testA*, *testB*, *trainA*, *trainB* with corresponding images.
  - Create folders named as:
    - *code/v5_cycleGAN/Pictures/Test/dataset_name/testA/*
    - *code/v5_cycleGAN/Pictures/Test/dataset_name/testB/*
    - *code/v5_cycleGAN/Pictures/Training/dataset_name/*
    - *models/v5_CycleGAN/*
  - Modify dataset variable in *code/v5_cycleGAN/train.py* and *code/v5_cycleGAN/test.py* to *dataset_name*.
  - Run *code/v5_cycleGAN/train.py*.
    - It saves losses in *code/v5_cycleGAN/loss_arr/* and logs in *code/v5_cycleGAN/loss_log/*.
    - It saves random generated images after each epoch in *code/v5_cycleGAN/Pictures/Training/dataset_name/*.
    - It saves model in *models/v5_CycleGAN/*.
 - Run *code/v5_cycleGAN/test.py*.
    - It saves generated images in *code/v5_cycleGAN/Pictures/Test/dataset_name/*.
 - Run *code/v5_cycleGAN/plot.py* for a training loss plot.
 
