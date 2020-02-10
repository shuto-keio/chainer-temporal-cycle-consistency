# chainer-temporal-cycle-consistency

This is an implementation of Temporal Cycle Consistency with Chainer.

The original project is [here](https://github.com/google-research/google-research/tree/master/tcc)



# Training, Evaluation and Visualization of t-SNE
1. To prepare the dataset, download download.sh from this [website](https://sites.google.com/site/brainrobotdata/home/multiview-pouring) and run dataset_preparation.sh

2. Start training. python train.py --device "your GPU device_id" -o "your output directory name"