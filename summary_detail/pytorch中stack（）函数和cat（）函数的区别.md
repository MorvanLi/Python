## pytorch中stack（）函数和cat（）函数的区别

在ResNet的[batch_predict](https://github.com/MorvanLi/Python/blob/main/pytorch_classification/ResNet/batch_predict.py)文件中，其中，batch_img = torch.stack(img_list, dim=0) （ 将img_list列表中的所有图像打包成一个batch），那么为什么不使用cat?二者的区别在于，**stack**（）函数是将两个输入堆叠起来，维度会增加1。而**cat**（）函数是将两个输入在现有维度上叠加起来，不改变维度数量。

```python
import torch
x = torch.rand((2,2,3))
y = torch.rand((2,2,3))
z = torch.stack((x,y),dim=0)
# 这样两个堆叠起来的 z 的维度是（2，2，2，3），相当于在dim=0将二者堆叠起来，第一个数字2就是增加的维度。
```



```python
import torch
x1 = torch.rand((2,2,3))
y1 = torch.rand((2,2,3))
z1 = torch.cat((x,y),dim=0)
# 这样两个叠加起来的 z1 的维度是（4，2，3），相当于在dim=0将二者叠加起来，不改变输入的维度。
```