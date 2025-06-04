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

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(os.listdir(root_dir))[:2048]

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
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
# 设置数据集目录和转换
# rgb_dir = "/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo/images/visible/train"
# thermal_dir = "/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo/images/infrared/train"
rgb_dir = "/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/visible/train"
thermal_dir = "/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/infrared/train"

rgb_dataset = CustomDataset(root_dir=rgb_dir, transform=transform)
thermal_dataset = CustomDataset(root_dir=thermal_dir, transform=transform)

targets = [0] * len(rgb_dataset) + [1] * len(thermal_dataset)
targets = np.asarray(targets).reshape(-1, 1)

# 创建dataloader
rgb_dataloader = DataLoader(rgb_dataset, batch_size=512, shuffle=False)
thermal_dataloader = DataLoader(thermal_dataset, batch_size=512, shuffle=False)

# 加载ResNet50模型
# resnet50 = torchvision.models.resnet50(pretrained=True)
# resnet50.eval().cuda()

# 加载ResNet50_zxModifiedStem
# config_file = '../runs/train/FLIR_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_flir_kdMed2Ear.py'
config_file = '../runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py'
cfg = Config.fromfile(config_file)
model = nn.Module()
model.backbone = MODELS.build(cfg['model']['backbone'])
load_checkpoint(model, os.path.join(os.path.dirname(config_file), 'epoch_12.pth'), map_location='cpu')
resnet50 = model.backbone
avgpool = nn.AdaptiveAvgPool2d((1, 1))
for p in resnet50.parameters():
    p.requires_grad = False
resnet50.cuda()

# 提取特征
def extract_features(dataloader):
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            outputs = resnet50(torch.cat([batch.cuda(), batch.cuda()], 1))[-1]  # RGB-RGB; T-T
            outputs = avgpool(outputs)
            outputs = torch.flatten(outputs, 1)
            # outputs = resnet50(batch.cuda())  # RGB; T
            features.append(outputs.detach().cpu())
    return torch.cat(features)

rgb_features = extract_features(rgb_dataloader)
thermal_features = extract_features(thermal_dataloader)

# 合并特征
merged_features = torch.cat((rgb_features, thermal_features), dim=0)

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(merged_features)

tsne_df = pd.DataFrame(
np.column_stack((tsne_features, targets)),
columns=["x", "y", "targets"]
)
tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
grid = sns.FacetGrid(tsne_df, hue="targets", height=10, palette=['#FF8B8B', '#61BFAD'])
grid.map(plt.scatter, "x", "y")
plt.axis('off')
for j in range(0, 2048, 400):
    plt.text(tsne_df.loc[:, "x"][j], tsne_df.loc[:, "y"][j], str(j))
    plt.text(tsne_df.loc[:, "x"][2048+j], tsne_df.loc[:, "y"][2048+j], str(2048+j))
    print(rgb_dataset.image_paths[j])
    print(thermal_dataset.image_paths[j])
plt.subplots_adjust(0.0, 0.0, 1.0, 1.0)
plt.show()
