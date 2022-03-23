# 如何使用卷积求梯度信息



```python
from scipy import signal as sg
import numpy as np
import torch
import torch.nn.functional as f
c = np.random.rand(5, 5)
d = c
c = np.pad(c, (0, 1))

print(c.shape)
r_shift_kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
b_shift_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

gradient = np.zeros([5, 5])
for x in range(5):
    for y in range(5):
        gx = abs(c[x + 1, y] - c[x, y])
        gy = abs(c[x, y + 1] - c[x, y])
        gradient[x, y] = gx + gy

print(gradient)
f1_r_shift = sg.convolve2d(d, r_shift_kernel, mode="same")
f1_b_shift = sg.convolve2d(d, b_shift_kernel, mode="same")

print(abs(f1_r_shift-d) + abs(f1_b_shift-d))

>>>[[0.62281698 0.59262771 0.79318934 0.91793342 0.13894122]
 [0.23996191 0.91838173 0.17099557 0.44575837 0.57312825]
 [1.06397267 0.55221546 0.77639971 0.41664271 0.61583979]
 [1.05962945 0.45494247 1.18330963 1.1964235  0.91762029]
 [0.45778835 0.842223   1.43174429 0.36219028 0.28642627]]

>>>[[0.62281698 0.59262771 0.79318934 0.91793342 0.13894122]
 [0.23996191 0.91838173 0.17099557 0.44575837 0.57312825]
 [1.06397267 0.55221546 0.77639971 0.41664271 0.61583979]
 [1.05962945 0.45494247 1.18330963 1.1964235  0.91762029]
 [0.45778835 0.842223   1.43174429 0.36219028 0.28642627]]
```



