#!/bin/bash
source /home/zqzhou/disk/miniconda3/bin/activate learn
python train.py \
--epochs 200 \
--batch_size 8 \
--arch Unet \
"--gpu id" 3 \
--data_dir .. \
--dataset VOC2011 \
--ratio 2 \
--input_channels 3 \
--num_classes 21 \
--input_w 240 \
--input_h 320 \
--optimizer Adam \
--lr 1e-4 \
--weight_decay 0 \
--momentum 0.9 \



