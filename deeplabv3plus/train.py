import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from option import Options
import encoding.utils as utils
from encoding.utils.metrics import *
from encoding.models import get_segmentation_model
from encoding.utils.lr_scheduler import LR_Scheduler
from encoding.datasets import get_segmentation_dataset
from encoding.utils.utils import load_checkpoint, AverageMeter, str2bool, set_seed


# ARCH_NAMES = archs.__all__

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.logger = utils.create_logger(self.args.exp_dir, "log")
        for k, v in vars(self.args).items():
            self.logger.info((k, v))

        if self.args.cuda:
            device = torch.device("cuda")
            self.logger.info("training on gpu:" + self.args.gpu_id)
        else:
            self.logger.info("training on cpu")
            device = torch.device("cpu")
        self.device = device

        #指定随机数
        set_seed(args.random_seed)
        
        # args.log_name = str(args.checkname)

        #好像是可以加速
        cudnn.benchmark = True

        #读取数据集，现在只有VOC
        self.logger.info('training on dataset '+ self.args.dataset)

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        data_kwargs = {'transform': input_transform, 'root':self.args.data_dir, 'base_size': self.args.base_size,
                    'crop_size': self.args.crop_size, 'logger': self.logger, 'scale': self.args.scale}

        trainset = get_segmentation_dataset(self.args.dataset, split='train', mode='train',
                                            **data_kwargs)
        testset = get_segmentation_dataset(self.args.dataset, split='val', mode='val',
                                            **data_kwargs)
        # dataloader
        kwargs = {'num_workers': self.args.num_workers, 'pin_memory': True} \
            if self.args.cuda else {}
        self.train_iter = data.DataLoader(trainset, batch_size=self.args.batch_size,
                                            drop_last=True, shuffle=True, **kwargs)
        self.val_iter = data.DataLoader(testset, batch_size=self.args.batch_size,
                                            drop_last=False, shuffle=False, **kwargs)
        self.num_classes = trainset.num_classes
        self.input_channels = trainset.input_channels

        #create model
        kwargs = {'fuse_attention':self.args.fuse_attention}
        self.model = get_segmentation_model(args.arch, dataset=args.dataset,backbone=args.backbone)
        print("=> creating model %s" % self.args.arch)
        # self.model = archs.__dict__[self.args.arch](num_classes=self.num_classes,
        # input_channels=self.input_channels, **model_kwargs)
        self.model = self.model.to(device)

        # self.logger.info(self.model)

        self.optimizer = None
        params = filter(lambda  p: p.requires_grad, self.model.parameters())
        if self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                params, lr=self.args.lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay
            )
        else:
            raise NotImplementedError

        #loss函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=trainset.IGNORE_INDEX)

        #语义分割评价指标
        self.metric = SegmentationMetric(self.num_classes)
        #学习率策略
        self.scheduler = LR_Scheduler(self.args.scheduler, base_lr = self.args.lr, num_epochs=self.args.epochs, \
            iters_per_epoch=len(self.train_iter))


        #创建实验结果保存目录
        self.writer = SummaryWriter(args.exp_dir)
        # with open(os.path.join(args.exp_dir,'config.yml'), 'w') as f:
        #     yaml.dump(config, f)

        #用tensoboard看一下模型结构
        X, label = next(iter(self.train_iter))
        self.writer.add_graph(self.model, X.to(device))
        
        self.epoch_begin = 0
        self.best_iou = 0.0

        #在训练开始前看看输出是什么
        val_log = self.validate(epoch=-1, is_visualize_segmentation=True)
        self.write_into_tensorboard(val_log, val_log, epoch=-1)

        #checkpoint_PATH
        if self.args.checkpoint_PATH is not None:
            if self.args.only_read_model:
                model, _, _, _, _ = load_checkpoint(model, self.args.checkpoint_PATH)
            else:
                model, self.epoch_begin, self.best_iou, self.optimizer= load_checkpoint(model, self.args.checkpoint_PATH, epoch_begin,  best_iou, optimizer, scheduler)

    def write_into_tensorboard(self, train_log, val_log, epoch):

            self.writer.add_scalars('0_Loss', {"train":train_log['loss'], "val":val_log['loss']}, epoch)
            self.writer.add_scalar('0_Loss/LR', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalars('1_mIoU', {"train":train_log['iou'], "val":val_log['iou']}, epoch)
            self.writer.add_scalar("1_mIoU/best_iou", self.best_iou, epoch)
            self.writer.add_scalars('2_Acc/acc', {"train":train_log['acc'], "val":val_log['acc']}, epoch)

    def training(self):
        #下面正式开始训练
        for epoch in range(self.epoch_begin, self.args.epochs):
            print('Epoch [%d/%d]' % (epoch, self.args.epochs))
            # train for one epoch
            train_log = self.train_one_epoch(epoch)
            val_log = self.validate(epoch, is_visualize_segmentation=True)

                    
            self.logger.info('epoch %d: lr %0.3e -  loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

            if val_log['iou'] >self.best_iou:
                self.best_iou = val_log['iou']
                # torch.save({'epoch':epoch, 'state_dict':model.state_dict(), 'best_iou':best_iou,
                # 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict()}, os.path.join(exp_dir,'model.pth'))
                torch.save({'epoch':epoch, 'state_dict':self.model.state_dict(), 'best_iou':self.best_iou,
                'optimizer':self.optimizer.state_dict()}, os.path.join(self.args.exp_dir,'model.pth'))
                print("=> saved best model")

            self.write_into_tensorboard(train_log, val_log, epoch)
            torch.cuda.empty_cache()

    def train_one_epoch(self, epoch):
        avg_meters = {'loss':AverageMeter()}

        self.metric.reset()
        self.model.train()
        pbar = tqdm(total=len(self.train_iter))
        for i, (X, labels) in enumerate(self.train_iter):
            self.scheduler(self.optimizer, i, epoch, self.best_iou)
            X = X.to(self.device)
            labels = labels.to(self.device)
            scores = self.model(X)
            loss = self.criterion(scores, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.metric.update(labels, scores)
                avg_meters['loss'].update(loss.item(), X.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        pixAcc, mIoU = self.metric.get()

        return OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', mIoU),
                ('acc', pixAcc)
            ])

    def validate(self, epoch, is_visualize_segmentation=True):
        avg_meters = {'loss':AverageMeter() }

        self.metric.reset()
        self.model.eval()
        visualizations  = []
        with torch.no_grad():
            pbar = tqdm(total=len(self.val_iter))
            for X, labels in self.val_iter:
                X = X.to(self.device)
                labels = labels.to(self.device)
                scores = self.model(X)
                loss = self.criterion(scores, labels)
                self.metric.update(labels, scores)

                avg_meters['loss'].update(loss.item(), X.size(0))

                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg)
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()

        pixAcc, mIoU = self.metric.get()

        self.visualize_segmentation(epoch)
        return OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', mIoU),
                ('acc', pixAcc)
            ])

    def visualize_segmentation(self, epoch):

        test_imgs_dir = self.args.test_imgs_dir
        self.model.eval()
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
                img_PIL = img_PIL.resize((self.args.crop_size, self.args.crop_size), Image.NEAREST)
                label_PIL = label_PIL.resize((self.args.crop_size, self.args.crop_size), Image.NEAREST)

                img_tensor = transforms.ToTensor()(img_PIL)
                img_tensor= transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)

                score = self.model(img_tensor.reshape(1, *img_tensor.shape).to(self.device)).squeeze().cpu()
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

        #原始图片只需要写一次
        if epoch == -1:
            self.writer.add_images("0_True/original_img", np.stack(imgs, 0), epoch, dataformats="NHWC")
            self.writer.add_images("0_True/imgs_label", np.stack(imgs_label, 0), epoch, dataformats="NHWC")
        self.writer.add_images("1_Predict/imgs_predict", np.stack(imgs_predict, 0), epoch, dataformats="NHWC")
        self.writer.add_images("1_Predict/imgs_score_map", np.stack(imgs_score_map, 0), epoch, dataformats="NHWC")

if __name__ == '__main__':
    args = Options().parse()
    trainer = Trainer(args)
    trainer.logger.info({'Total Epochs:', str(args.epochs)})
    trainer.training()