#__author:Administrator  
#date: 2022/3/11
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
import os
_pil_gray = transforms.ToPILImage()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("加载的数据不只只是数据，还有很多，即：")
# print(variables['y'])
print('内部变量为：')


# plt.figure(figsize=(25, 25))
# for i in range(64):
#     ax = plt.subplot(8, 8, i+1)
#     # [H, W, C]
#     plt.imshow(variables['img1'][:, :, i])
# plt.show()


device = 'cuda'
_tensor = transforms.ToTensor()
model = DenseFuseNet().to(device)
model.load_state_dict(torch.load('./train_result/model_weight.pkl')['weight'])

for i in range(13):
    variables = io.loadmat(f'./fusion/fusion{i+1}.mat')
    result = model.decoder(_tensor(variables['y']).unsqueeze(0).to(device))
    result = result.squeeze(0)
    img_fusion = transforms.ToPILImage()(result)
    img_fusion.save(f'./result/{i+1}.png')
    # img_fusion.show()