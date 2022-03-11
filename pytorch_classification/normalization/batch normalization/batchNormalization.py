# author_='bo.li';
# date: 3/3/22 5:58 PM

import numpy as np
import torch.nn as nn
import torch


def bn_process(feature, mean, var):
    feature_shape = feature.shape
    for i in range(feature_shape[1]):
        # [batch, channel, height, width]
        feature_t = feature[:, i, :, :]
        mean_t = feature_t.mean()
        # 总体标准差
        std_t1 = feature_t.std()
        # 样本标准差
        std_t2 = feature_t.std(ddof=1)

        # bn process
        # 这里记得加上eps和pytorch保持一致
        feature[:, i, :, :] = (feature[:, i, :, :] - mean_t) / np.sqrt(std_t1 ** 2 + 1e-5)

        # update calculating mean and var
        mean[i] = mean[i] * 0.9 + mean_t * 0.1
        var[i] = var[i] * 0.9 + (std_t2 ** 2) * 0.1
    print(feature)


# 随机生成一个batch为2，channel为2，height=width=2的特征向量
# [batch, channel, height, width]
feature1 = torch.randn(2, 2, 2, 2)
# 初始化统计均值和方差
calculate_mean = [0.0, 0.0]
calculate_var = [1.0, 1.0]
# print(feature1.numpy())

# 注意要使用copy()深拷贝
bn_process(feature1.numpy().copy(), calculate_mean, calculate_var)

bn = nn.BatchNorm2d(2, eps=1e-5)
output = bn(feature1)
print(output)


import torch
import torch.nn as nn


def layer_norm_process(feature: torch.Tensor, beta=0., gamma=1., eps=1e-5):
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
    # 均值
    mean = var_mean[1]
    # 方差
    var = var_mean[0]

    # layer norm process
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature


def main():
    t = torch.rand(4, 2, 3)
    print(t)
    # 仅在最后一个维度上做norm处理
    norm = nn.LayerNorm(normalized_shape=t.shape[-1], eps=1e-5)
    # 官方layer norm处理
    t1 = norm(t)
    # 自己实现的layer norm处理
    t2 = layer_norm_process(t, eps=1e-5)
    print("t1:\n", t1)
    print("t2:\n", t2)


if __name__ == '__main__':
    main()
