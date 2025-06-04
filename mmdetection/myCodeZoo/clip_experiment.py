import numpy as np
import torch
import clip
from PIL import Image
import os.path as osp
from torchvision.models.resnet import resnet50
import torch.nn as nn
from glob import glob
import torch.nn.functional as F
from mmdet.models.detectors.retinanet_clip import unfold_batch

classes = ['person', 'car', 'bus', 'motorcycle', 'traffic light', 'truck']
num_class = len(classes)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root='./')
# model, preprocess = clip.load("RN50", device=device, download_root='./')
root = '/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/visible/train'
img_list = sorted(glob(osp.join(root, '*.png')))
rgb_img = Image.open(img_list[0])
t_img = Image.open(img_list[0].replace('visible', 'infrared'))
rgb_image = preprocess(rgb_img).unsqueeze(0).to(device)
t_image = preprocess(t_img).unsqueeze(0).to(device)
batch_inputs = torch.cat([rgb_image, t_image], 1)
text_inputs = torch.cat(
    [clip.tokenize(f"a photo of the {c}") for c in classes + ['others']]).to(device)

# CLIP
## global
batch_global_inputs = F.interpolate(batch_inputs, clip_model.visual.input_resolution, mode='bicubic')
with torch.no_grad():
    batch_global_rgb_inputs, batch_global_t_inputs = torch.chunk(batch_global_inputs, 2, 1)
    global_rgb_feats = clip_model.encode_image(batch_global_rgb_inputs)  # [b, d]
    global_t_feats = clip_model.encode_image(batch_global_t_inputs)  # [b, d]
    text_features = clip_model.encode_text(text_inputs)  # [c+1, d]
global_rgb_feats /= global_rgb_feats.norm(dim=-1, keepdim=True)
global_t_feats /= global_t_feats.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
global_rgb_similarity = (100.0 * global_rgb_feats @ text_features.T).softmax(dim=-1)  # [b, c+1]
global_t_similarity = (100.0 * global_t_feats @ text_features.T).softmax(dim=-1)  # [b, c+1]
global_similarity = (global_rgb_similarity + global_t_similarity) / 2.0  # [b, c+1]
## local
(batch_patches_rgb_inputs, batch_patches_t_inputs), patch_shape = unfold_batch(batch_inputs,
                                                                               clip_model.visual.input_resolution)
with torch.no_grad():
    local_rgb_feats, local_t_feats = clip_model.encode_image(
        batch_patches_rgb_inputs), clip_model.encode_image(batch_patches_t_inputs)
local_rgb_feats /= local_rgb_feats.norm(dim=-1, keepdim=True)
local_t_feats /= local_t_feats.norm(dim=-1, keepdim=True)

rgb_similarity = (100.0 * local_rgb_feats @ text_features.T).softmax(dim=-1)  # [b*nBlock, c+1]
t_similarity = (100.0 * local_t_feats @ text_features.T).softmax(dim=-1)  # [b*nBlock, c+1]
rgb_similarity = rgb_similarity.view(patch_shape[0], patch_shape[1], num_class + 1)  # [b, nBlock, c+1]
t_similarity = t_similarity.view(patch_shape[0], patch_shape[1], num_class + 1)  # [b, nBlock, c+1]
local_similarity = torch.maximum(rgb_similarity, t_similarity)  # [b, nBlock, c+1]
# local_similarity = F.normalize(local_similarity, 1, dim=-1)    # [b, nBlock, c+1]
max_patch_similarity = local_similarity.max(dim=1)[0]  # [b, c+1]
min_patch_similarity = local_similarity.min(dim=1)[0]  # [b, c+1]
gamma = torch.where(max_patch_similarity.detach() > 0.5, 1.0, 0.0)  # [b, c+1]
local_similarity = gamma * max_patch_similarity + (1.0 - gamma) * min_patch_similarity  # [b, c+1]
clip_scores = (local_similarity + global_similarity) / 2.0  # [b, c+1]

from myCodeZoo.show_feat import *
from torchvision.utils import make_grid

batch_patches_rgb_inputs_img = make_grid(batch_patches_rgb_inputs, nrow=9, normalize=True, scale_each=True).permute(1, 2, 0).detach().cpu().numpy()
batch_patches_t_inputs_img = make_grid(batch_patches_t_inputs, nrow=9, normalize=True, scale_each=True).permute(1, 2, 0).detach().cpu().numpy()

fig = plt.figure()
fig.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.imshow(np.vstack([batch_patches_rgb_inputs_img, batch_patches_t_inputs_img]))
plt.show()