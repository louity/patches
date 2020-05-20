for eps in 1e-1 1e-2 1e-3 1e-4 1e-5 0.
do
  python kneighbors.py --n_channel_convolution 2048 --batchsize 128 --dataset cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5 --spatialsize_convolution 6 --lr "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 128 --padding_mode reflect --kneighbors_fraction 0.4 --convolutional_classifier 6 --whitening_reg $eps --sgd_momentum 0.9 --batch_norm --summary_file ablation_study_eps.txt
done
