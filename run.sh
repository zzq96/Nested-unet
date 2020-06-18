#!/bin/bash
source /home/zqzhou/disk/miniconda3/bin/activate learn
python train.py \
--epochs 300 \
--batch_size 8 \
--arch NestedUnet \
--deep_supervision False \
"--gpu id" 0 \
--data_dir . \
--dataset VOC2011 \
--ratio 8 \
--input_channels 3 \
--num_classes 21 \
--input_h 256 \
--input_w 256 \
--optimizer Adam \
--lr 1e-2 \
--weight_decay 0 \
--momentum 0.9 \
--scheduler  ReduceLROnPlateau \
--lr_gamma 0.5 \
--patience 7 \
--min_lr 1e-6 \
--test_imgs_dir test_imgs \
--random_seed 1337 \
