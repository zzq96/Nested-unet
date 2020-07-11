#!/bin/bash
source /home/zqzhou/disk/miniconda3/bin/activate backupWithTFAndTensorboard
python train.py \
--epochs 500 \
--batch_size 8 \
--arch FCN32s \
--deep_supervision False \
"--gpu id" -1 \
--data_dir . \
--dataset VOC2012 \
--ratio 10 \
--input_channels 3 \
--num_classes 21 \
--input_h 256 \
--input_w 256 \
--optimizer SGD \
--lr 1e-2 \
--weight_decay 5e-4 \
--momentum 0.9 \
--scheduler  ReduceLROnPlateau \
--lr_gamma 0.5 \
--patience 10 \
--min_lr 5e-6 \
--test_imgs_dir test_imgs \
--random_seed 1337 \
