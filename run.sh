#!/bin/bash
#source /home/zqzhou/disk/miniconda3/bin/activate learn
python train.py \
--epochs 200 \
--batch_size 6 \
--arch NestedUnet \
"--gpu id" 0 \
--data_dir . \
--dataset VOC2011 \
--ratio 2 \
--input_channels 3 \
--num_classes 21 \
--input_h 224 \
--input_w 224 \
--optimizer Adam \
--lr 3e-4 \
--weight_decay 0 \
--momentum 0.9 \
--scheduler ReduceLROnPlateau \
--lr_gamma 0.5 \
--patience 4 \
--min_lr 1e-6 \
--test_imgs_dir test_imgs \
--random_seed 1337 \



