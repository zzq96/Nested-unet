import torch
import numpy as np
import argparse
from torch import nn
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import archs 
from utils import load_checkpoint,  load_data_VOCSegmentation, init_weights, get_upsampling_weight, AverageMeter,MultiRandomCrop, PILImageConcat, str2bool
from torch.optim import lr_scheduler
from loss import *
import sys
import time
import os
from tqdm import tqdm
import yaml
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from encoding.datasets import get_segmentation_dataset
from torchvision import transforms
from torch.utils import data

ARCH_NAMES = archs.__all__
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    
    # model
    parser.add_argument('--gpu id', default=None, type=int,
                         help='use which gpu, if id = -1, use cpu')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: Unet)')
    parser.add_argument('--fuse_attention', default=False, type=str2bool)
    parser.add_argument('--checkpoint_PATH', default=None)
    parser.add_argument('--input_channels', default=3, type=int,
                         help='input channels')
    parser.add_argument('--num_classes', default=21, type=int,
                        help='number of classes')
    parser.add_argument('--base_size', default=300, type=int,
                        help='image base_size')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='image crop size')
    parser.add_argument('--scale', default=True, type=str2bool)
    
    # loss
    # parser.add_argument('--loss', default='BCEDiceLoss',
    #                     choices=LOSS_NAMES,
    #                     help='loss: ' +
    #                     ' | '.join(LOSS_NAMES) +
    #                     ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--test_imgs_dir', default='.',
                        help='test imgs dir')
    parser.add_argument('--data_dir', default='.',
                        help='dataset dir')
    parser.add_argument('--dataset', default='voc2011',
                        help='dataset name')
    parser.add_argument('--ratio', default= 1, type=int,
                        help='only use 1/ratio\'s dataset')
    # parser.add_argument('--img_ext', default='.png',
                        # help='image file extension')
    # parser.add_argument('--mask_ext', default='.png',
                        # help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='ConstantLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'StepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    #parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--lr_gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--random_seed', default=0, type=int)

    config = parser.parse_args()

    return config

def train(config, train_iter, model, criterion, optimizer,device):
    avg_meters = {'loss':AverageMeter(),
    'iou':AverageMeter(), 
    'acc':AverageMeter(), 
    'acc_cls':AverageMeter() 
    }

    model.train()
    
    pbar = tqdm(total=len(train_iter))
    for X, labels in train_iter:
        X = X.to(device)
        labels = labels.to(device)
        scores = model(X)
        loss = criterion(scores, labels)
        acc, acc_cls, iou = iou_score(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), X.size(0))
        avg_meters['iou'].update(iou, X.size(0))
        avg_meters['acc'].update(acc, X.size(0))
        avg_meters['acc_cls'].update(acc_cls, X.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('acc', avg_meters['acc'].avg),
            ('acc_cls', avg_meters['acc_cls'].avg)
        ])

def validate(config, val_iter, model, criterion, device):
    avg_meters = {'loss':AverageMeter(),
    'iou':AverageMeter(),
    'acc':AverageMeter(), 
    'acc_cls':AverageMeter() 
    }

    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(total=len(val_iter))
        for X, labels in val_iter:
            X = X.to(device)
            labels = labels.to(device)
            scores = model(X)
            loss = criterion(scores, labels)
            acc, acc_cls, iou = iou_score(scores, labels)

            avg_meters['loss'].update(loss.item(), X.size(0))
            avg_meters['iou'].update(iou, X.size(0))
            avg_meters['acc'].update(acc, X.size(0))
            avg_meters['acc_cls'].update(acc_cls, X.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('acc', avg_meters['acc'].avg),
            ('acc_cls', avg_meters['acc_cls'].avg)
        ])

def predict(model, save_dir, epoch, config, device, writer):

    test_imgs_dir = config['test_imgs_dir']
    model.eval()
    imgs = []
    imgs_label = []
    imgs_predict = []
    imgs_score_map = []
    palette = None
    with torch.no_grad():
        for filename in os.listdir(test_imgs_dir):
            if 'jpg' not in filename:
                continue
            img_PIL = Image.open(os.path.join(test_imgs_dir, filename))#RGB模式
            label_PIL = Image.open(os.path.join(test_imgs_dir, filename.replace('jpg', 'png')))#P模式

            ##获取调色板    
            if palette == None:
                palette = label_PIL.getpalette()
            #用最近邻缩放图片
            img_PIL = img_PIL.resize((config['crop_size'], config['crop_size']), Image.NEAREST)
            label_PIL = label_PIL.resize((config['crop_size'], config['crop_size']), Image.NEAREST)

            img_tensor = torchvision.transforms.ToTensor()(img_PIL)
            img_tensor= torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)

            score = model(img_tensor.resize(1, *img_tensor.shape).to(device)).squeeze().cpu()
            score = torch.softmax(score, dim=0)
            pre = score.max(dim=0)
            label_pred = pre[1].data.numpy().astype(np.uint8) 
            label_pred = Image.fromarray(label_pred)
            label_pred.putpalette(palette)


            #label_pred.putpalette(label.getpalette())

            imgs.append(np.array(img_PIL))
            imgs_label.append(np.array(label_PIL.convert("RGB")))
            imgs_predict.append(np.array(label_pred.convert("RGB")))
            plt.imsave('tmp/score_map.png', score[0])
            imgs_score_map.append(plt.imread("tmp/score_map.png")[:, :, :3])

    if epoch == -1:
        writer.add_images("0_True/original_img", np.stack(imgs, 0), epoch, dataformats="NHWC")
        writer.add_images("0_True/imgs_label", np.stack(imgs_label, 0), epoch, dataformats="NHWC")
    writer.add_images("1_Predict/imgs_predict", np.stack(imgs_predict, 0), epoch, dataformats="NHWC")
    writer.add_images("1_Predict/imgs_score_map", np.stack(imgs_score_map, 0), epoch, dataformats="NHWC")

def main():
    config = vars(parse_args())

    np.random.seed(config['random_seed'])
    torch.random.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])

    if config['name'] is None:
        config['name'] = '%s_%s' % (config['arch'], config['dataset'])
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())[2:]
    #TODO:cur_time想想取什么目录名
    exp_dir = os.path.join(sys.path[0], 'exps',config['name'], cur_time + "_" + \
        config["optimizer"] + "_lr_" + '{:0.0e}_'.format(config["lr"])+"wd_"+
        '{:0.0e}'.format(config["weight_decay"]) + '_fa_' + str(config['fuse_attention']))
    print('-' * 20)
    for key in config:
        print('%s:%s' %(key, config[key]))
    print('-' * 20)
    
    
    # gpu_id == None，说明使用cpu
    if config['gpu id'] is not None and config['gpu id'] >=0:
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu id'])
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        device = torch.device("cpu")

    #好像是可以加速
    cudnn.benchmark = True


    #读取配置

    #读取数据集，现在只有VOC
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    data_kwargs = {'transform': input_transform, 'root':config['data_dir'], 'base_size': config["base_size"],
                    'crop_size': config["crop_size"], 'logger': None, 'scale': config["scale"]}

    trainset = get_segmentation_dataset(config["dataset"], split='train', mode='train',
                                        **data_kwargs)
    testset = get_segmentation_dataset(config["dataset"], split='val', mode='val',
                                        **data_kwargs)
    # dataloader
    kwargs = {'num_workers': config["num_workers"], 'pin_memory': True} \
        if config['gpu id'] >= 0 else {}
    train_iter = data.DataLoader(trainset, batch_size=config["batch_size"],
                                        drop_last=True, shuffle=True, **kwargs)
    val_iter = data.DataLoader(testset, batch_size=config["batch_size"],
                                        drop_last=False, shuffle=False, **kwargs)
    num_classes = trainset.num_class
            
    #累计梯度设置，1就是不累积
    #TODO:没考虑bn层的表现
    accumulation_steps = 1

    #create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](num_classes=num_classes,
    input_channels=config['input_channels'], fuse_attention=config['fuse_attention'])
    model = model.to(device)
    if config['checkpoint_PATH'] is not None:
        _, model, _, _, _ = load_checkpoint(model, config['checkpoint_PATH'])
    print("training on", device)

    params = filter(lambda  p: p.requires_grad, model.parameters())
    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay']
        )
    else:
        raise NotImplementedError


    #用于梯度累计的计数
    iter_cnt = 0

    #loss函数
    criterion = nn.CrossEntropyLoss()
    #学习率策略
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['lr_gamma'], patience=config['patience'],
                                                   verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['lr_gamma'])
    elif config['scheduler'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['lr_gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    #创建实验结果保存目录
    writer = SummaryWriter(exp_dir)
    with open(os.path.join(exp_dir,'config.yml'), 'w') as f:
        yaml.dump(config, f)

    X, label = next(iter(train_iter))
    writer.add_graph(model, X.to(device))
    #在训练开始前看看输出是什么
    epoch = -1
    predict(model, exp_dir, epoch, config, device, writer)
    val_log = validate(config, val_iter, model, criterion, device)
    writer.add_scalars('0_Loss', {"train":val_log['loss'], "val":val_log['loss']}, epoch)
    writer.add_scalars('1_mIoU', {"train":val_log['iou'], "val":val_log['iou']}, epoch)
    writer.add_scalar("1_mIoU/best_iou", val_log['iou'], epoch)

    writer.add_scalars('2_Acc_cls', {"train":val_log['acc_cls'], "val":val_log['acc_cls']}, epoch)
    writer.add_scalars('3_Acc', {"train":val_log['acc'], "val":val_log['acc']}, epoch)


    #下面正式开始训练
    best_iou = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        start_time = time.time()
        # train for one epoch
        train_log = train(config, train_iter, model, criterion, optimizer,device)
        val_log = validate(config, val_iter, model, criterion, device)

        if config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        elif config['scheduler'] == 'ConstantLR':
            pass
        else:
            scheduler.step()


        predict(model, exp_dir, epoch, config, device, writer)
                
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))


        if val_log['iou'] >best_iou:
            best_iou = val_log['iou']
            torch.save({'epoch':epoch, 'state_dict':model.state_dict(), 'best_iou':best_iou,
            'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict()}, os.path.join(exp_dir,'model.pth'))
            print("=> saved best model")

        # writer.add_scalar("Loss/train", train_log['loss'], epoch)
        # writer.add_scalar("Loss/val", val_log['loss'], epoch)
        #writer.add_scalar("mIoU/train", train_log['iou'], epoch)
        #writer.add_scalar("mIoU/val", val_log['iou'], epoch)
        writer.add_scalars('0_Loss', {"train":train_log['loss'], "val":val_log['loss']}, epoch)
        writer.add_scalars('1_mIoU', {"train":train_log['iou'], "val":val_log['iou']}, epoch)
        writer.add_scalar("1_mIoU/best_iou", best_iou, epoch)

        writer.add_scalars('2_Acc_cls', {"train":train_log['acc_cls'], "val":val_log['acc_cls']}, epoch)
        writer.add_scalars('3_Acc', {"train":train_log['acc'], "val":val_log['acc']}, epoch)
        # writer.add_scalar("Acc/train", train_log['acc'], epoch)
        # writer.add_scalar("Acc/val", val_log['acc'], epoch)
        # writer.add_scalar("Acc_cls/train", train_log['acc_cls'], epoch)
        # writer.add_scalar("Acc_cls/val", val_log['acc_cls'], epoch)

        torch.cuda.empty_cache()



if __name__ == '__main__':
    #config = vars(parse_args())
    #fcn = archs.FCN32s(21, 3)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #fcn.load_state_dict(torch.load('Nested-unet/fcn32.pth'))
    #fcn.to('cpu')
    #predict(fcn, 'Nested-unet/test_imgs', "Nested-unet/test", 0, config)
    main()

    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # train_iter, val_iter = load_data_VOCSegmentation(year="2011", batch_size=8, crop_size=(320, 480),\
    #     root='Datasets/VOC/',num_workers=4, use=4)

    # net = Unet(num_classes=21, in_channels=3)
    # net.apply(init_weights)

    # # net = FCN32s(21)

    # print(list(net.modules()), len(list(net.modules())))

    # #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3)
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # #optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # trainer(net, train_iter, val_iter, nn.CrossEntropyLoss(), optimizer, scheduler, num_epochs=100, gpu_id=3)