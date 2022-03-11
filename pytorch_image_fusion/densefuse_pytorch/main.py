# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:32:38 2019

@author: win10
"""
import torch
from PIL import Image
from densefuse_net import DenseFuseNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from scipy.io import savemat
from scipy import io


variables = io.loadmat('data.mat')
print("加载的数据不只只是数据，还有很多，即：")
# print(variables)
print('内部变量为：')
a = variables['img1'][:, :, 1]

plt.figure(figsize=(25, 25))
for i in range(64):
    ax = plt.subplot(8, 8, i+1)
    # [H, W, C]
    plt.imshow(variables['img1'][:, :, i])
plt.show()

# print(variables['score_of_jerry'])









device = 'cuda'
_tensor = transforms.ToTensor()

model = DenseFuseNet().to(device)
model.load_state_dict(torch.load('./train_result/model_weight.pkl')['weight'])
img = Image.open('./1.tif')
img = _tensor(img).unsqueeze(0)
features = model.encoder(img.to(device))
for feature_map in features:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.cpu().detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
    savemat('data.mat', {f'img1': im})
    # show top 12 feature maps
    plt.figure(figsize=(25, 25))
    for i in range(64):
        ax = plt.subplot(8, 8, i+1)
        # [H, W, C]
        plt.imshow(im[:, :, i])
    plt.show()

# test_path = './images/IV_images/'
# test(test_path, model, mode='add')