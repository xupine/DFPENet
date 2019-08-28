import argparse
import os
import numpy as np
from tqdm import tqdm
import torch

from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.DFPENet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary


import matplotlib
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt

from PIL import Image
import matplotlib.image as mping
import cv2
from skimage import io
from dataloaders.utils import decode_segmap
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def nor(self,img):

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class DFPE_test(object):
    """docstring for DFPE"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    INPUT_SIZE = 600
    cuda = torch.cuda.is_available()
    gpu_ids = [0]
    def __init__(self,model_path):
        self.lr = 0.007
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        if DFPE_test.cuda and len(DFPE_test.gpu_ids) > 1:
            self.sync_bn = True
        else:
            self.sync_bn = False
        self.model = DFPENet(num_classes=6,backbone='resnet',output_stride=16,sync_bn=self.sync_bn,freeze_bn=False)
        self.train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        self.optimizer = torch.optim.SGD(self.train_params, momentum=0.9,weight_decay=5e-4, nesterov=False)
        if DFPE_test.cuda:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=DFPE_test.gpu_ids)
            patch_replication_callback(self.model)
        #if DFPE_test.cuda:
        checkpoint = torch.load(model_path)
        #else:
            #device = torch.device("cpu")
            #checkpoint = torch.load(model_path,map_location=device)
        epoch = checkpoint['epoch']
        if DFPE_test.cuda:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:

            self.model.load_state_dict(checkpoint['state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_pred = checkpoint['best_pred']
    def nor(self,img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img
    def run(self,TEST_IMAGE_PATHS):
        self.model.eval()
        for image_path in TEST_IMAGE_PATHS:
            img = Image.open(image_path).convert('RGB')
            img=self.nor(img)
            img = img.transpose((2, 0, 1))



            print(img.shape)
            img = img[np.newaxis, :, :, :]
            image = torch.from_numpy(img)

            if DFPE_test.cuda:
                image = image.cuda()

            with torch.no_grad():
                seg_map = self.model(image)
            seg_map = seg_map.data.cpu().numpy()
            seg_map = np.argmax(seg_map, axis=1)
            print(seg_map.shape)
        return seg_map
    def vis_segmentation(self,image, seg_map):
        plt.figure(figsize=(30, 50))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
        plt.subplot(grid_spec[0])
        plt.imshow(image)
        plt.axis('off')
        plt.title('input image')
        plt.subplot(grid_spec[1])

        seg_map = seg_map.squeeze()
        seg_map = np.array(seg_map).astype(np.uint8)
        seg_image = decode_segmap(seg_map,dataset='vaihingen')
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('segmentation map')

        plt.subplot(grid_spec[2])
        plt.imshow(image)
        plt.imshow(seg_image, alpha=0.7)
        plt.axis('off')
        plt.title('segmentation overlay')


        #plt.show()
        #seg_image = seg_image.astype(np.float32)
        maxi = seg_image.max()
        seg_image = seg_image/maxi*255
        seg_image = seg_image.astype(np.uint8)
        #image = Image.fromarray(image)
        seg_image=Image.fromarray(seg_image)
        #images=Image.blend(image,seg_image,0.4)
        seg_image.save('D:\\Segmentation\\test_image\\DFPEPD\\image5.png')

        #matplotlib.image.imsave('D:\\Segmentation\\test\\seg_image.tif',seg_image)

model_path ='D:\\Segmentation\\model\\model_best_DFPNPD.pth.tar'
MODEL = DFPE_test(model_path)
PATH_TO_TEST_IMAGES_DIR = 'D:\\Segmentation\\test_image\\src'
r = 5
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.png'.format(i)) for i in range(r,r+1) ]
seg_maps = MODEL.run(TEST_IMAGE_PATHS)
for image_path in TEST_IMAGE_PATHS:
    mying=Image.open(image_path)
    mying=mying.convert('RGB')
    mying=np.array(mying, dtype=np.uint8)
    print(mying.shape)
MODEL.vis_segmentation(mying, seg_maps)

















        
        