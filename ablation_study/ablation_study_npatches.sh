for n_patches in 512 1024 2048 4096
do
  python kneighbors.py --n_channel_convolution $n_patches --batchsize 128 --dataset cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5 --spatialsize_convolution 6 --lr "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 128 --padding_mode reflect --kneighbors_fraction 0.4 --convolutional_classifier 6 --whitening_reg 1e-3 --sgd_momentum 0.9 --batch_norm --summary_file ablation_study_npatches.txt
done
