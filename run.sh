#!/bin/bash
source /home/zqzhou/disk/miniconda3/bin/activate learn
python train.py \
--epochs 500 \
--batch_size 16 \
--arch FCN32s \
--deep_supervision False \
"--gpu id" 0 \
--data_dir Datasets \
--dataset vocaug \
--ratio 1 \
--input_channels 3 \
--base_size 300 \
--crop_size 256 \
--scale True \
--optimizer SGD \
--lr 1e-2 \
--weight_decay 5e-4 \
--momentum 0.9 \
--scheduler  ReduceLROnPlateau \
--lr_gamma 0.5 \
--patience 15 \
--min_lr 5e-6 \
--test_imgs_dir test_imgs \
--random_seed 1337 \
