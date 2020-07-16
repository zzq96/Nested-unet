###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2018
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')

        parser.add_argument('--exp_name', default=None,
                            help='model name: (default: arch+timestamp)')
        # model
        parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                            choices=ARCH_NAMES,
                            help='model architecture: ' +
                            ' | '.join(ARCH_NAMES) +
                            ' (default: Unet)')

        #dataset
        parser.add_argument('--test_imgs_dir', default='.',
                            help='test imgs dir')
        parser.add_argument('--data_dir', default='.',
                            help='dataset dir')
        parser.add_argument('--dataset', default='voc2011',
                            help='dataset name')
        parser.add_argument('--ratio', default= 1, type=int,
                            help='only use 1/ratio\'s dataset')
        parser.add_argument('--base_size', default=300, type=int,
                            help='image base_size')
        parser.add_argument('--crop_size', default=224, type=int,
                            help='image crop size')
        parser.add_argument('--use_scale', default=True, type=str2bool)
        parser.add_argument('--num_workers', default=4, type=int)

        # training hyper params
        parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('-b', '--batch_size', default=8, type=int,
                            metavar='N', help='mini-batch size (default: 8)')
        # parser.add_argument('--loss', default='BCEDiceLoss',
        #                     choices=LOSS_NAMES,
        #                     help='loss: ' +
        #                     ' | '.join(LOSS_NAMES) +
        #                     ' (default: BCEDiceLoss)')
        
        # optimizer
        parser.add_argument('--optimizer', default='SGD',
                            choices=['Adam', 'SGD'])
        parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='momentum')
        parser.add_argument('--weight_decay', default=0, type=float,
                            help='weight decay')
        parser.add_argument('--nesterov', default=False, type=str2bool,
                            help='nesterov')
        
        # lr scheduler
        parser.add_argument('--lr_scheduler', default='Poly',
                            choices=['ReduceLROnPlateau', 'StepLR', 'ConstantLR', 'Poly'])
        parser.add_argument('--min_lr', default=1e-6, type=float,
                            help='minimum learning rate')
        parser.add_argument('--lr_step', type=int, default=None)
        #parser.add_argument('--factor', default=0.1, type=float)
        parser.add_argument('--patience', default=5, type=int)
        # parser.add_argument('--milestones', default='1,2', type=str)
        parser.add_argument('--lr_gamma', default=0.5, type=float)
        # parser.add_argument('--early_stopping', default=-1, type=int,
        #                     metavar='N', help='early stopping (default: -1)')

        
        # cuda, seed and logging
        parser.add_argument('--cuda', default=False, type=str2bool)
        parser.add_argument('--gpu id', default=None, type=str)
        parser.add_argument('--random_seed', default=1337, type=int)
        parser.add_argument('--log_root', type=str, default=None)

        # checking point
        parser.add_argument('--checkpoint_PATH', default=None)
        parser.add_argument('--only_read_model', default=False, type=str2bool)

        #custom
        parser.add_argument('--fuse_attention', default=False, type=str2bool)

        # parser.add_argument('--input_channels', default=3, type=int,
        #                     help='input channels')
        # parser.add_argument('--num_classes', default=21, type=int,
        #                     help='number of classes')
        
        
        # parser.add_argument('--img_ext', default='.png',
                            # help='image file extension')
        # parser.add_argument('--mask_ext', default='.png',
                            # help='mask file extension')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = args.cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'voc2011': 250,
                'voc2012': 250,
                'vocaug': 250,
                'pcontext': 80,
                'ade20k': 160,
                'cityscapes': 300,
            }
            args.epochs = epoches[args.dataset.lower()]
        # if args.batch_size is None:
        #     args.batch_size = 4 * torch.cuda.device_count()
        # if args.test_batch_size is None:
        #     args.test_batch_size = args.batch_size
        # if args.lr is None:
        #     lrs = {
        #         'pascal_voc': 0.0001,
        #         'pascal_aug': 0.001,
        #         'pcontext': 0.001,
        #         'ade20k': 0.01,
        #         'cityscapes': 0.01,
        #     }
        #     args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
        return args
