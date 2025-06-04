import numpy as np
import torch
import copy
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import os
from tqdm import tqdm
import cv2
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import torch.nn as nn
os.environ['DISPLAY'] = 'localhost:10.0'

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.image_paths = sorted(os.listdir(root_dir))[:2048]
        self.image_paths = sorted(os.listdir(root_dir))[1599:1600]

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
dataset_name = 'FLIR'

if dataset_name == 'FLIR':
    rgb_dir = "/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo/images/visible/train"
    thermal_dir = "/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo/images/infrared/train"
else:
    rgb_dir = f"/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/visible/train"
    thermal_dir = f"/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/infrared/train"

rgb_dataset = CustomDataset(root_dir=rgb_dir, transform=rgb_transform)
thermal_dataset = CustomDataset(root_dir=thermal_dir, transform=t_transform)

save_path = f'./thesisCh4_featureVis_{dataset_name}/{os.path.basename(rgb_dataset.image_paths[0]).split(".")[0]}'
os.makedirs(save_path, exist_ok=True)

# 创建dataloader
rgb_dataloader = DataLoader(rgb_dataset, batch_size=512, shuffle=False)
thermal_dataloader = DataLoader(thermal_dataset, batch_size=512, shuffle=False)

# 加载ResNet50模型
# resnet50 = torchvision.models.resnet50(pretrained=True)
# resnet50.eval().cuda()

# 加载ResNet50_zxModifiedStem
if dataset_name == "FLIR":
    config_file_wWeakSup = '../runs/train/FLIR_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_flir_kdMed2Ear.py'
    config_file_woWeakSup = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/FLIR_rgbtEarly_gfl_r50_fpn_1x_bs4/gfl_r50_fpn_1x_flir.py'
else:
    config_file_wWeakSup = '../runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py'
    config_file_woWeakSup = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_rgbtEarly_gfl_r50_fpn_1x_bs4/gfl_r50_fpn_1x_m3fd.py'
cfg_wWeakSup = Config.fromfile(config_file_wWeakSup)
cfg_woWeakSup = Config.fromfile(config_file_woWeakSup)
# with weakly supervised learning
res50_wWeakSup = nn.Module()
res50_wWeakSup.backbone = MODELS.build(cfg_wWeakSup['model']['backbone'])
load_checkpoint(res50_wWeakSup, os.path.join(os.path.dirname(config_file_wWeakSup), 'epoch_12.pth'), map_location='cpu')
res50_wWeakSup = res50_wWeakSup.backbone
for p in res50_wWeakSup.parameters():
    p.requires_grad = False
res50_wWeakSup.cuda()

# without weakly supervised learning
res50_woWeakSup = nn.Module()
res50_woWeakSup.backbone = MODELS.build(cfg_woWeakSup['model']['backbone'])
load_checkpoint(res50_woWeakSup, os.path.join(os.path.dirname(config_file_woWeakSup), 'epoch_12.pth'), map_location='cpu')
res50_woWeakSup = res50_woWeakSup.backbone
for p in res50_woWeakSup.parameters():
    p.requires_grad = False
res50_woWeakSup.cuda()

# 提取特征
def extract_features(dataloader, net):
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            outputs = net(torch.cat([batch.cuda(), batch.cuda()], 1))  # RGB-RGB; T-T
            features = [out.detach().cpu() for out in outputs]
    return features

rgb_features = extract_features(rgb_dataloader, net=res50_wWeakSup)
thermal_features = extract_features(thermal_dataloader, net=res50_wWeakSup)
rgb_features_woSup = extract_features(rgb_dataloader, net=res50_woWeakSup)
thermal_features_woSup = extract_features(thermal_dataloader, net=res50_woWeakSup)

_, _, h, w = rgb_features[0].shape
rgb_outs = []
t_outs = []
rgb_outs_wosup = []
t_outs_wosup = []
for rgbf, tf, rgbf_wosup, tf_wosup in zip(rgb_features, thermal_features, rgb_features_woSup, thermal_features_woSup):
    # clamp_value = 5.0
    # rgbf = rgbf.clamp_max(clamp_value)
    # tf = tf.clamp_max(clamp_value)
    # rgbf_wosup = rgbf_wosup.clamp_max(clamp_value)
    # tf_wosup = tf_wosup.clamp_max(clamp_value)
    tau = 2
    rgb_outs.append(F.interpolate(((rgbf/tau).softmax(1) * rgbf).sum(1, keepdim=True), (h, w), mode='bilinear'))
    t_outs.append(F.interpolate(((tf/tau).softmax(1) * tf).sum(1, keepdim=True), (h, w), mode='bilinear'))
    rgb_outs_wosup.append(F.interpolate(((rgbf_wosup/tau).softmax(1) * rgbf_wosup).sum(1, keepdim=True), (h, w), mode='bilinear'))
    t_outs_wosup.append(F.interpolate(((tf_wosup/tau).softmax(1) * tf_wosup).sum(1, keepdim=True), (h, w), mode='bilinear'))

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

rgb_outs = torch.stack(rgb_outs)
t_outs = torch.stack(t_outs)
rgb_outs = rgb_outs.sum(0).squeeze().detach().numpy()
t_outs = t_outs.sum(0).squeeze().detach().numpy()

rgb_outs_wosup = torch.stack(rgb_outs_wosup)
t_outs_wosup = torch.stack(t_outs_wosup)
rgb_outs_wosup = rgb_outs_wosup.sum(0).squeeze().detach().numpy()
t_outs_wosup = t_outs_wosup.sum(0).squeeze().detach().numpy()

H, W = rgb_dataloader.dataset[0].shape[1:]
cv2.imwrite(f'{save_path}/rgb_wWeaklySup.png',
            cv2.resize(cv2.applyColorMap((normalize(rgb_outs) * 255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H),
                       interpolation=cv2.INTER_LINEAR))
cv2.imwrite(f'{save_path}/t_wWeaklySup.png',
            cv2.resize(cv2.applyColorMap((normalize(t_outs) * 255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H),
                       interpolation=cv2.INTER_LINEAR))
cv2.imwrite(f'{save_path}/rgb_woWeaklySup.png',
            cv2.resize(cv2.applyColorMap((normalize(rgb_outs_wosup) * 255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H),
                       interpolation=cv2.INTER_LINEAR))
cv2.imwrite(f'{save_path}/t_woWeaklySup.png',
            cv2.resize(cv2.applyColorMap((normalize(t_outs_wosup) * 255.).astype('uint8'), cv2.COLORMAP_VIRIDIS), (W, H),
                       interpolation=cv2.INTER_LINEAR))



