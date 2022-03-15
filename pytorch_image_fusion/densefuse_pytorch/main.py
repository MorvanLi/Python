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

# variables = io.loadmat('fusion.mat')
# print("加载的数据不只只是数据，还有很多，即：")
# print(variables['y'])
# print('内部变量为：')


# plt.figure(figsize=(25, 25))
# for i in range(64):
#     ax = plt.subplot(8, 8, i+1)
#     # [H, W, C]
#     plt.imshow(variables['img1'][:, :, i])
# plt.show()

# print(variables['score_of_jerry'])









device = 'cuda'
_tensor = transforms.ToTensor()

model = DenseFuseNet().to(device)
model.load_state_dict(torch.load('./train_result/model_weight.pkl')['weight'])
for i in range(13):
    img1 = Image.open(f'./images/c{i+1}_1.tif')
    img1 = _tensor(img1).unsqueeze(0)

    img2 = Image.open(f'./images/c{i+1}_2.tif')
    img2 = _tensor(img2).unsqueeze(0)
    features1 = model.encoder(img1.to(device))
    features2 = model.encoder(img2.to(device))

    # savemat(f'data{i}.mat', {'img': im})

    for feature_map1, feature_map2 in zip(features1, features2):
        # [N, C, H, W] -> [C, H, W]
        im1 = np.squeeze(feature_map1.cpu().detach().numpy())
        im2 = np.squeeze(feature_map2.cpu().detach().numpy())
        # [C, H, W] -> [H, W, C]
        im1 = np.transpose(im1, [1, 2, 0])
        im2 = np.transpose(im2, [1, 2, 0])
        savemat(f'./matfiles/data{i+1}_1.mat', {'img': im1})
        savemat(f'./matfiles/data{i+1}_2.mat', {'img': im2})
        # # show top 12 feature maps
        # plt.figure(figsize=(25, 25))
        # for i in range(64):
        #     ax = plt.subplot(8, 8, i+1)
        #     # [H, W, C]
        #     plt.imshow(im[:, :, i])
        # plt.show()







# result = model.decoder(_tensor(variables['y']).unsqueeze(0).to(device))
# result = result.squeeze(0)
# img_fusion = transforms.ToPILImage()(result)
# img_fusion.show()




# plt.figure(figsize=(25, 25))
# for i in range(64):
#     ax = plt.subplot(8, 8, i+1)
#     # [H, W, C]
#     plt.imshow(variables['y'][:, :, i])
# plt.show()






# test_path = './images/IV_images/'
# test(test_path, model, mode='add')