# patches

## CIFAR 10

### Linear regression

  * Radius Neighborhood encoding for whitening cosine distance
    * 2K patches
      ```
      python radiusneighbors.py --num_workers 4 --n_channel_convolution 2048 --batchsize 128 --dataset cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5  --lr "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 64 --padding_mode reflect --shrink heaviside --convolutional_classifier 6 --whitening_reg 1e-4 --bias 0.07 --sgd_momentum 0.9 --batch_norm
      ```

    * 16K patches
    ```
    python radiusneighbors.py --num_workers 4 --n_channel_convolution 16384 --batchsize 128 --dataset  cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5  --lr "{0:3e-3,100:3e-4,150:3e-5}" --nepochs 175 --optimizer SGD --bottleneck_dim 128 --padding_mode reflect --shrink heaviside --convolutional_classifier 6 --whitening_reg 1e-4 --bias 0.07 --sgd_momentum 0.9 --batch_norm
    ```

  * K-nearest neighbors for whitening linear distance
    * 2K patches
      ```
      python kneighbors.py --n_channel_convolution 2048 --batchsize 128 --dataset cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5 --finalsize_avg_pooling 6 --lr "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 64 --padding_mode reflect --kneighbors_fraction 0.4 --convolutional_classifier 6 --whitening_reg 1e-3 --sgd_momentum 0.9 --batch_norm
      ```
    * 16K patches
    ```
    python kneighbors.py --num_workers 4 --n_channel_convolution 16384 --batchsize 128 --dataset cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5  --lr "{0:3e-3,100:3e-4,150:3e-5}" --nepochs 175 --optimizer SGD --bottleneck_dim 64 --padding_mode reflect  --kneighbors_fraction 0.4 --convolutional_classifier 6 --whitening_reg 1e-3 --sgd_momentum 0.9 --batch_norm
    ```
    Best test acc. 85.62% at epoch 158/174, final test acc 85.5%

### 1 hidden layer classifier


## Imagenet 64

### Linear regression

  * K-nearest neighbors for whitening linear distance
    * 2K patches
      ```
      python kneighbors.py --num_workers 4 --n_channel_convolution 2048 --batchsize 256 --dataset imagenet64 --path_train ~/datasets/imagenet64/out_data_train --path_test ~/datasets/imagenet64/out_data_val --spatialsize_convolution 6  --stride_avg_pooling 3 --spatialsize_avg_pooling 5  --lr "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 192 --padding_mode reflect --kneighbors_fraction 0.4 --convolutional_classifier 12 --whitening_reg 1e-3 --sgd_momentum 0.9 --batch_norm
      ```
      Best test acc. 33.21 top1 (54.4 top5) at epoch 79/79.
