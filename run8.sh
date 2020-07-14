#!/bin/bash
source /home/zqzhou/disk/miniconda3/bin/activate learn
python train.py \
--epochs 500 \
--batch_size 16 \
--arch FCN8s \
--fuse_attention True \
--checkpoint_PATH exps/FCN8s_vocaug/200714_151112_SGD_lr_1e-03_wd_0e+00_fa_True/model.pth \
"--gpu id" 1 \
--data_dir Datasets \
--dataset vocaug \
--ratio 1 \
--input_channels 3 \
--base_size 360 \
--crop_size 256 \
--scale True \
--optimizer SGD \
--lr 1e-4 \
--weight_decay 0 \
--momentum 0.9 \
--scheduler  ReduceLROnPlateau \
--lr_gamma 0.5 \
--patience 5 \
--min_lr 5e-6 \
--test_imgs_dir test_imgs \
--random_seed 1337 \
