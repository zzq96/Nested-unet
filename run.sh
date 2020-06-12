#!/bin/bash
source /home/zqzhou/disk/miniconda3/bin/activate learn
python train.py \
--epochs 200 \
--batch_size 20 \
--arch Unet \
"--gpu id" 3 \
--data_dir .. \
--dataset VOC2011 \
--ratio 8 \
--input_channels 3 \
--num_classes 21 \
--input_w 224 \
--input_h 224 \
--optimizer SGD \
--lr 1e-3 \
--weight_decay 5e-4 \
--momentum 0.9 \



