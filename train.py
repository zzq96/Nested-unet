import torch
import argparse
from torch import nn
import archs 
from archs import *
from utils import load_data_VOCSegmentation, init_weights, get_upsampling_weight
from torch.optim import lr_scheduler
from loss import *
def trainer(net, train_iter, val_iter, loss_f, optimizer, scheduler, num_epochs, gpu_id = 0):

    accumulation_steps = 1
    # gpu_id == None，说明使用cpu
    device = torch.device("cuda" if gpu_id != None else 'cpu')
    if gpu_id:
        import os
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

if __name__ == "__main__":
     torch.manual_seed(0)
     torch.cuda.manual_seed(0)
     train_iter, val_iter = load_data_VOCSegmentation(year="2011", batch_size=4, crop_size=(320, 480),\
         root='Datasets/VOC/',num_workers=4, use=4)

     net = Unet(num_classes=21, in_channels=3)
     net.apply(init_weights)

     # net = FCN32s(21)

     print(list(net.modules()), len(list(net.modules())))

     #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3)
     optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
     #optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
     trainer(net, train_iter, val_iter, nn.CrossEntropyLoss(), optimizer, scheduler, num_epochs=100, gpu_id=2)