###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2018
###########################################################################

import os
import argparse
import torch
from encoding.utils.utils import str2bool
import time, sys

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')

        parser.add_argument('--exp_dir', default=None)
        # model
        parser.add_argument('--arch', '-a', type=str)
        parser.add_argument('--backbone', default='resnet50', type=str)

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
        parser.add_argument('--scale', default=True, type=str2bool)
        parser.add_argument('--num_workers', default=16, type=int)

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
        parser.add_argument('--scheduler', default='Poly',
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
        parser.add_argument('--gpu_id', default=None, type=str)
        parser.add_argument('--random_seed', default=1337, type=int)
        parser.add_argument('--log_root', type=str, default=None)

        # checking point
        parser.add_argument('--ft', default=False, type=str2bool)
        parser.add_argument('--resume', default=False, type=str2bool)
        parser.add_argument('--checkpoint_PATH', default=None)

        #custom
        parser.add_argument('--fuse_attention', default=False, type=str2bool)

        
        # parser.add_argument('--img_ext', default='.png',
                            # help='image file extension')
        # parser.add_argument('--mask_ext', default='.png',
                            # help='mask file extension')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id #必须要放在所有用到cuda的代码前，下面的cuda.manual_seed也是
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

        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())[2:]
        if args.exp_dir is None:
            args.exp_dir = os.path.join(sys.path[0], '.' + args.dataset, cur_time + "_" + \
                args.optimizer + "_lr_" + '{:0.0e}_'.format(args.lr)+"wd_"+
                '{:0.0e}'.format(args.weight_decay) + '_fa_' + str(args.fuse_attention))
        print(args)
        # print('-' * 20)
        # for key in config:
        #     print('%s:%s' %(key, args.key))
        # print('-' * 20)
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
