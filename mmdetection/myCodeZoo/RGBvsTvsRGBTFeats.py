'''
为了验证weakly supervised learning能够消除RGB-T之间的domain gap问题，
这里使用ImageNet预训练的ResNet50提取RGB和Thermal的特征，
对比使用weakly supervised leaning获得的ResNet50得到的RGB-T的特征

重点比较对目标特征的提取能力，因为我们这里提到的domain gap问题主要针对
目标特征的提取能力
'''
import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import seaborn as sns
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import torch.nn as nn
import torch.nn.functional as F
from myCodeZoo.show_feat import *
# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.image_paths = sorted(os.listdir(root_dir))[:2048]
        self.image_paths = [os.path.join(root_dir, '00227.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 加载数据集
rgb_transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(mean=[149.4/255., 148.7/255., 141.7/255.], std=[49.3/255., 52.8/255., 59.0/255.])  # 标准化
])
t_transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(mean=[135.7/255., 135.7/255., 135.7/255.], std=[63.6/255., 63.6/255., 63.6/255.])  # 标准化
])
# 设置数据集目录和转换
# rgb_dir = "/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo/images/visible/train"
# thermal_dir = "/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo/images/infrared/train"
rgb_dir = "/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/visible/test"
thermal_dir = "/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/infrared/test"

rgb_dataset = CustomDataset(root_dir=rgb_dir, transform=rgb_transform)
thermal_dataset = CustomDataset(root_dir=thermal_dir, transform=t_transform)

targets = [0] * len(rgb_dataset) + [1] * len(thermal_dataset)
targets = np.asarray(targets).reshape(-1, 1)

# 创建dataloader
rgb_dataloader = DataLoader(rgb_dataset, batch_size=8, shuffle=False)
thermal_dataloader = DataLoader(thermal_dataset, batch_size=8, shuffle=False)

# 加载ResNet50模型
rgb_model = nn.Module()
t_model = nn.Module()
rgb_model.backbone = MODELS.build(dict(type='ResNet', depth=50, init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained')))
t_model.backbone = MODELS.build(dict(type='ResNet', depth=50, init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained')))
load_checkpoint(rgb_model,
                '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_rgb_gfl_r50_fpn_1x_bs4/epoch_12.pth')
load_checkpoint(t_model,
                '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_thermal_gfl_r50_fpn_1x_bs4/epoch_12.pth')
rgb_resnet50 = rgb_model.backbone
t_resnet50 = t_model.backbone
rgb_resnet50.init_weights()
t_resnet50.init_weights()
rgb_resnet50.cuda()
t_resnet50.cuda()

# 加载ResNet50_zxModifiedStem
# config_file = '../runs/train/FLIR_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_flir_kdMed2Ear.py'
config_file = '../runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py'
config_file1 = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_rgbtEarly_gfl_r50_fpn_1x_bs4/gfl_r50_fpn_1x_m3fd.py'
cfg = Config.fromfile(config_file)
cfg1 = Config.fromfile(config_file1)
model = nn.Module()
model1 = nn.Module()
model.backbone = MODELS.build(cfg['model']['backbone'])
model1.backbone = MODELS.build(cfg1['model']['backbone'])
load_checkpoint(model, os.path.join(os.path.dirname(config_file), 'epoch_12.pth'), map_location='cpu')
rgbt_backbone = model.backbone
rgbt_backbone1 = model1.backbone
rgbt_backbone1.init_weights()
rgbt_backbone.cuda()
rgbt_backbone1.cuda()

# RGB-T提取特征
cv2.namedWindow('rgbt')
# cv2.namedWindow('rgb')
# cv2.namedWindow('t')
for rgb, t in tqdm(zip(rgb_dataloader, thermal_dataloader), total=len(rgb_dataloader)):
    # RGB-T
    with torch.no_grad():
        outputs = rgbt_backbone(torch.cat([rgb.cuda(), t.cuda()], 1))  # RGB-RGB; T-T
        outputs1 = rgbt_backbone1(torch.cat([rgb.cuda(), t.cuda()], 1))  # RGB-RGB; T-T
        _, _, h, w = outputs[0].shape
        rgbt_outs = []
        rgbt_outs1 = []
        for tmp, tmp1 in zip(outputs, outputs1):
            rgbt_outs.append(F.interpolate((tmp.softmax(1) * tmp).sum(1, keepdim=True), (h, w), mode='bilinear'))
            rgbt_outs1.append(F.normalize(F.interpolate((tmp1.softmax(1) * tmp1).sum(1, keepdim=True), (h, w), mode='bilinear'), dim=(2, 3)))
        rgbt_outs = torch.stack(rgbt_outs)
        rgbt_outs1 = torch.stack(rgbt_outs1)
        rgbt_outs = rgbt_outs.sum(0)
        rgbt_outs1 = rgbt_outs1.sum(0)
        rgbt_outs = rgbt_outs.clamp_max(3.0)
        rgbt_outs1 = rgbt_outs1.clamp_max(0.08)
        rgbt_outs_img = torchvision.utils.make_grid(rgbt_outs, 8, padding=0, scale_each=True, normalize=True).permute(1, 2, 0).squeeze().detach().cpu().numpy()
        rgbt_outs_img1 = torchvision.utils.make_grid(rgbt_outs1, 8, padding=0, scale_each=True, normalize=True).permute(1, 2, 0).squeeze().detach().cpu().numpy()

    # RGB
    with torch.no_grad():
        rgb_outputs = rgb_resnet50(rgb.cuda())
        t_outputs = t_resnet50(t.cuda())

        rgb_outs, t_outs = [], []
        for rgb_tmp, t_tmp in zip(rgb_outputs, t_outputs):
            rgb_outs.append(F.normalize(F.interpolate((rgb_tmp.softmax(1) * rgb_tmp).sum(1, keepdim=True), (h, w), mode='bilinear'), dim=(2, 3)))
            t_outs.append(F.normalize(F.interpolate((t_tmp.softmax(1) * t_tmp).sum(1, keepdim=True), (h, w), mode='bilinear'), dim=(2, 3)))
        rgb_outs = torch.stack(rgb_outs)
        t_outs = torch.stack(t_outs)
        rgb_outs = rgb_outs.sum(0).clamp_max(0.1)
        t_outs = t_outs.sum(0).clamp_max(0.1)
        rgb_outs_img = torchvision.utils.make_grid(rgb_outs, 8, padding=0, scale_each=True, normalize=True).permute(1, 2, 0).squeeze().detach().cpu().numpy()
        t_outs_img = torchvision.utils.make_grid(t_outs, 8, padding=0, scale_each=True, normalize=True).permute(1, 2, 0).squeeze().detach().cpu().numpy()
        # cv2.imshow('rgbt', np.vstack([np.hstack([rgbt_outs_img, rgbt_outs_img1]), np.hstack([rgb_outs_img, t_outs_img])]))
        cv2.imshow('rgbt', np.vstack([np.hstack([cv2.applyColorMap((x*255.).astype('uint8'), cv2.COLORMAP_VIRIDIS) for x in [rgbt_outs_img[..., 0], rgbt_outs_img1[..., 0], rgb_outs_img[..., 0], t_outs_img[..., 0]]])]))
        _, _, H, W = rgb.shape
        cv2.imwrite('./rgb.png', cv2.resize(cv2.applyColorMap((rgb_outs_img[..., 0]*255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H), interpolation=cv2.INTER_LINEAR))
        cv2.imwrite('./thermal.png', cv2.resize(cv2.applyColorMap((t_outs_img[..., 0]*255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H), interpolation=cv2.INTER_LINEAR))
        cv2.imwrite('./rgbt_woWeak.png', cv2.resize(cv2.applyColorMap((rgbt_outs_img1[..., 0]*255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H), interpolation=cv2.INTER_LINEAR))
        cv2.imwrite('./rgbt_wWeak.png', cv2.resize(cv2.applyColorMap((rgbt_outs_img[..., 0]*255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H), interpolation=cv2.INTER_LINEAR))
        # cv2.imshow('rgb', np.hstack([rgb_outs_img, t_outs_img]))
        # cv2.imshow('t', t_outs_img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
cv2.destroyAllWindows()



