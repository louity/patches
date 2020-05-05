# patches


Example of command :
```
python metric.py --num_workers 4 --n_channel_convolution 2048 --batchsize 128 --dataset cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5  --lr "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 64 --padding_mode reflect --shrink heaviside --convolutional_classifier 6 --whitening_reg 1e-4 --bias 0.07 --sgd_momentum 0.9 --normalize_net_outputs
```
