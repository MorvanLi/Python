# author_='bo.li';
# date: 3/7/22 10:47 AM

import torch
import torch.nn as nn


def group_norm(x: torch.Tensor,
               num_groups: int,
               num_channels: int,
               eps: float = 1e-5,
               gramma: float = 1.0,
               beta: float = 0.):
    """

    :param x: input data must a tensor
    :param num_groups:
    :param num_channels:
    :param eps: default value is 1e-5
    :param gramma: default value is 1.
    :param beta: default value is 0.
    return a tensor with same shape x
    """
    assert divmod(num_channels, num_groups)[1] == 0
    channels_per_group = num_channels // num_groups

    new_tensor = []
    # [b, c, h, w] ----> ([b, 2, h, w], [b, 2, h, w])
    for c in x.split(channels_per_group, dim=1):
        var_mean = torch.var_mean(c, dim=[1, 2, 3], unbiased=False)
        var = var_mean[0]
        mean = var_mean[1]
        c = (c-mean[:, None, None, None]) / (torch.sqrt(var[:, None, None, None] + eps))
        c = c * gramma + beta
        new_tensor.append(c)
    new_tensor = torch.cat(new_tensor, dim=1)
    return new_tensor

def main():
    num_groups = 2
    num_channels = 4
    eps = 1e-5

    img = torch.rand(2, num_channels, 2, 2)
    # print(img)

    gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
    r1 = gn(img)
    print(r1)

    r2 = group_norm(img, num_groups, num_channels, eps)
    print(r2)


if __name__ == '__main__':
    main()