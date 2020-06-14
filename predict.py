import torch
from torch import nn
import matplotlib.pylab as plt
from archs import *
import os
from PIL import Image
import torchvision

def predict(model, test_imgs_dir, save_dir, epoch, config, device):

    #test_imgs_dir = config['test_imgs_dir']
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        print("test_imgs_dir:%s" % test_imgs_dir)
        cnt = 0
        for filename in os.listdir(test_imgs_dir):
            if 'jpg' not in filename:
                continue
            img = Image.open(os.path.join(test_imgs_dir, filename))#RGB模式
            label = Image.open(os.path.join(test_imgs_dir, filename.replace('jpg', 'png')))#P模式

            #用最近邻缩放图片
            img = img.resize((config['input_w'], config['input_h']), Image.NEAREST)
            label = label.resize((config['input_w'], config['input_h']), Image.NEAREST)

            img = torchvision.transforms.ToTensor()(img)
            img = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            score = model(img.resize(1, *img.shape).to(device)).squeeze()
            for i in range(21):
                plt.imsave(save_dir+'/'+str(i)+ ' ' + str(cnt)+'.png', score[i])
            #pre = score.max(dim=0)
            #label_pred = pre[1].data.cpu().numpy().astype(np.uint8) 
            #label_pred = Image.fromarray(label_pred)

            #label_pred.putpalette(label.getpalette())
            #new_img = PILImageConcat(label, label_pred)
            #new_img.putpalette(label.getpalette())
            #new_img.save(save_dir+'/'+str(epoch)+ ' ' + str(cnt)+'.png')
            cnt += 1
if __name__ == "__main__":

    config = {'input_w':224, 'input_h':224}
    unet = Unet(21, 3) 
    unet.load_state_dict(torch.load(r'exps\Unet_VOC2011\2020-06-13_07.51.10\model.pth'))
    unet.to('cpu')
    predict(unet, 'test_imgs', "test", -1, config, 'cpu')
