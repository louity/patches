for kneighbors in 10  50 100 500 800 1000 1500
do
  python kneighbors.py --n_channel_convolution 2048 --batchsize 128 --dataset cifar10 --stride_avg_pooling 3 --spatialsize_avg_pooling 5 --spatialsize_convolution 6 --lr "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 128 --padding_mode reflect --kneighbors $kneighbors --convolutional_classifier 6 --whitening_reg 1e-3 --sgd_momentum 0.9 --batch_norm --summary_file ablation_study_K.txt
done
