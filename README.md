# Generative Models with Discrete Representation Learning


## Intro

This repository implements training VQVAE model and training latent prior distribution.

This repository implements parts of the paper, [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (VQ-VAE).


## Requirements
- python3.5
- Numpy
- tensorflow 1.4
- tqdm



### train models

- Train models on MNIST dataset: 'python main.py'
which is default as 
python $SRCDIR/main.py --data_set=mnist --train_num=60000 --K=4 --D=128 --grad_clip=1.0 --num_feature_maps=32

- Train mdoels on CIFAR10 dataset: 
python $SRCDIR/main.py --data_set=cifar10 --train_num=200000 --K=10 --D=256 --grad_clip=5.0 --num_feature_maps=64



## Future work: 

- Implement WaveNet and possibly ByteNet into the framework.

- Improve sampling speed



## Acknowledgement
- Parts of the codes are adopted from [hiwonjoon's repo](https://github.com/hiwonjoon/tf-vqvae and https://github.com/anantzoid/Conditional-PixelCNN-decoder) and [anantzoid's repo.](https://github.com/anantzoid/Conditional-PixelCNN-decoder), especially, hiwonjoon's prior distribution training work. 
