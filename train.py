import torch
import argparse
from torch import nn
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import archs 
from utils import load_data_VOCSegmentation, init_weights, get_upsampling_weight, AverageMeter
from torch.optim import lr_scheduler
from loss import *
import sys
import time
import os
from tqdm import tqdm
import yaml
import pandas as pd

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
    # parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                         help='input channels')
    parser.add_argument('--num_classes', default=21, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=224, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=224, type=int,
                        help='image height')
    
    # loss
    # parser.add_argument('--loss', default='BCEDiceLoss',
    #                     choices=LOSS_NAMES,
    #                     help='loss: ' +
    #                     ' | '.join(LOSS_NAMES) +
    #                     ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--data_dir', default='.',
                        help='dataset name')
    parser.add_argument('--dataset', default='VOC2011',
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
    # parser.add_argument('--nesterov', default=False, type=str2bool,
                        # help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default=None,
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def trainer(net, train_iter, val_iter, loss_f, optimizer, scheduler, num_epochs, gpu_id = 0):

    accumulation_steps = 1
    # gpu_id == None，说明使用cpu
    device = torch.device("cuda" if gpu_id != None else 'cpu')
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    net = net.to(device)
    print("training on", device)
    epoch_cnt = 0

    with torch.no_grad():
        net.eval()
        train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc = evaluate_accuracy(train_iter, net, loss_f, device)
        val_loss, val_acc, val_acc_cls,val_mean_iu, val_fwavacc = evaluate_accuracy(val_iter, net, loss_f, device)
        print("epoch: begin")
        print("train_loss: %f, train_acc: %f, train_acc_cls:%f, train_mean_iu:%f, train_fwavacc:%f" % (train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc))
        print("val_loss: %f, val_acc: %f, val_acc_cls:%f, val_mean_iu:%f, val_fwavacc:%f" % (val_loss, val_acc, val_acc_cls, val_mean_iu, val_fwavacc))

    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()
        for X, labels in train_iter:
            epoch_cnt += 1

            X = X.to(device)
            labels = labels.to(device)
            scores = net(X)
            loss = loss_f(scores, labels)
            #print("loss",loss.cpu().item())
            loss = loss/accumulation_steps
            loss.backward()
            if epoch_cnt % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                #print("lr", optimizer.param_groups[0]['lr'])
        scheduler.step()
                

        with torch.no_grad():
            net.eval()
            train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc = evaluate_accuracy(train_iter, net, loss_f, device)
            val_loss, val_acc, val_acc_cls,val_mean_iu, val_fwavacc = evaluate_accuracy(val_iter, net, loss_f, device)
            print("epoch: %d, time: %d sec" % (epoch + 1, time.time() - start_time))
            print("lr", optimizer.param_groups[0]['lr'])
            print("train_loss: %f, train_acc: %f, train_acc_cls:%f, train_mean_iu:%f, train_fwavacc:%f" % (train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc))
            print("val_loss: %f, val_acc: %f, val_acc_cls:%f, val_mean_iu:%f, val_fwavacc:%f" % (val_loss, val_acc, val_acc_cls, val_mean_iu, val_fwavacc))


def train(config, train_iter, model, criterion, optimizer,device):
    avg_meters = {'loss':AverageMeter(),
    'iou':AverageMeter()
    }

    model.train()
    
    pbar = tqdm(total=len(train_iter))
    for X, labels in train_iter:
        X = X.to(device)
        labels = labels.to(device)
        scores = model(X)
        loss = criterion(scores, labels)
        iou = iou_score(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), X.size(0))
        avg_meters['iou'].update(iou, X.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg)
        ])

def validate(config, val_iter, model, criterion, device):
    avg_meters = {'loss':AverageMeter(),
    'iou':AverageMeter()
    }

    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(total=len(val_iter))
        for X, labels in val_iter:
            X = X.to(device)
            labels = labels.to(device)
            scores = model(X)
            loss = criterion(scores, labels)
            iou = iou_score(scores, labels)

            avg_meters['loss'].update(loss.item(), X.size(0))
            avg_meters['iou'].update(iou, X.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg)
        ])


def main():
    config = vars(parse_args())
    if config['name'] is None:
        config['name'] = '%s_%s' % (config['arch'], config['dataset'])
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    exp_dir = os.path.join(sys.path[0], 'exps',config['name'], cur_time)
    os.makedirs(exp_dir, exist_ok=True)
    print('-' * 20)
    for key in config:
        print('%s:%s' %(key, config[key]))
    print('-' * 20)
    
    with open(os.path.join(exp_dir,'config.yml'), 'w') as f:
        yaml.dump(config, f)
    
    # define loss function

    #好像是可以加速
    cudnn.benchmark = True

    #create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
    config['input_channels'])
    model.apply(init_weights)

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

    #读取数据集，现在只有VOC
    if 'VOC' in config['dataset']:
        if '2011' in config['dataset']:
            train_iter, val_iter = load_data_VOCSegmentation(year="2011", batch_size=config['batch_size'], \
                    crop_size=(config['input_h'], config['input_w']), \
                    root=os.path.join(config['data_dir'],'Datasets/VOC/'),num_workers=config['num_workers'], use=config['ratio'])
        elif '2012' in config['dataset']:
            train_iter, val_iter = load_data_VOCSegmentation(year="2012", batch_size=config['batch_size'], \
                    crop_size=(config['input_h'], config['input_w']), \
                    root=os.path.join(config['data_dir'],'Datasets/VOC/'),num_workers=config['num_workers'], use=config['ratio'])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    log =  OrderedDict([
        ('epoch',[]),
        ('lr',[]),
        ('train_loss',[]),
        ('train_iou',[]),
        ('val_loss',[]),
        ('val_iou',[]),
        ('best_iou', []),
        ('time', []),
    ])

    best_iou = 0
            
    #累计梯度设置，1就是不累积
    #TODO:没考虑bn层的表现
    accumulation_steps = 1
    # gpu_id == None，说明使用cpu
    device = torch.device("cuda" if config['gpu id'] != None and config['gpu id'] > 0 else 'cpu')
    if config['gpu id']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu id'])

    model = model.to(device)
    print("training on", device)


    #用于梯度累计的计数
    iter_cnt = 0

    #loss函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        start_time = time.time()
        # train for one epoch
        train_log = train(config, train_iter, model, criterion, optimizer,device)
        val_log = validate(config, val_iter, model, criterion, device)


        if config['scheduler'] is not None:
            raise NotImplementedError
                
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))


        if val_log['iou'] >best_iou:
            torch.save(model.state_dict(), os.path.join(exp_dir,'model.pth'))
            best_iou = val_log['iou']
            print("=> saved best model")

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['train_loss'].append(train_log['loss'])
        log['train_iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['best_iou'].append(best_iou)
        log['time'].append(time.time() - start_time)

        pd.DataFrame(log).to_csv(os.path.join(exp_dir,'log.csv'), index=False)
        
        torch.cuda.empty_cache()

        ##验证集上测试
        # with torch.no_grad():
            # model.eval()
            # train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc = evaluate_accuracy(train_iter, model, criterion, device)
            # val_loss, val_acc, val_acc_cls,val_mean_iu, val_fwavacc = evaluate_accuracy(val_iter, model, criterion, device)
            # print("epoch: %d, time: %d sec" % (epoch + 1, time.time() - start_time))
            # print("lr", optimizer.param_groups[0]['lr'])
            # print("train_loss: %f, train_acc: %f, train_acc_cls:%f, train_mean_iu:%f, train_fwavacc:%f" % (train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc))
            # print("val_loss: %f, val_acc: %f, val_acc_cls:%f, val_mean_iu:%f, val_fwavacc:%f" % (val_loss, val_acc, val_acc_cls, val_mean_iu, val_fwavacc))


if __name__ == '__main__':
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