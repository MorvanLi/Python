

## pytorch中stack（）函数和cat（）函数的区别

Q1: 在ResNet的[batch_predict](https://github.com/MorvanLi/Python/blob/main/pytorch_classification/ResNet/batch_predict.py)文件中，其中，batch_img = torch.stack(img_list, dim=0) （ 将img_list列表中的所有图像打包成一个batch），那么为什么不使用cat?

A:  二者的区别在于，**stack**（）函数是将两个输入堆叠起来，维度会增加1。而**cat**（）函数是将两个输入在现有维度上叠加起来，不改变维度数量。

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



## Tensorflow2.1 GPU安装与Pytorch1.3 GPU安装

参考我之前写的博文：[Centos7 安装Tensorflow2.1 GPU以及Pytorch1.3 GPU（CUDA10.1）](https://blog.csdn.net/qq_37541097/article/details/103933366)


## keras functional api训练的模型权重与subclassed训练的模型权重能否混用 [tensorflow2.0.0]
强烈不建议混用，即使两个模型的名称结构完全一致也不要混用，里面有坑，用什么方法训练的模型就载入相应的模型权重


## 使用subclassed模型时无法使用model.summary() [tensorflow2.0.0]
subclassed模型在实例化时没有自动进行build操作（只有在开始训练时，才会自动进行build），如果需要使用summary操作，需要提前手动build  
model.build((batch_size, height, width, channel))


## 无法使用keras的plot_model(model, 'my_model.png')问题 [tensorflow2.0.0]
#### 在linux下你需要安装一些包：
* pip install pydot==1.2.3
* sudo apt-get install graphviz   
#### 在windows中，同样需要安装一些包（windows比较麻烦）：
* pip install pydot==1.2.3
* 安装graphviz，并添加相关环境变量  
参考连接：https://github.com/XifengGuo/CapsNet-Keras/issues/7

## 为什么每计算一个batch，就需要调用一次optimizer.zero_grad() [Pytorch1.3]   
如果不清除历史梯度，就会对计算的历史梯度进行累加（通过这个特性你能够变相实现一个很大batch数值的训练）   
参考链接：https://www.zhihu.com/question/303070254    

## Pytorch1.3 ImportError: cannot import name 'PILLOW_VERSION' [Pytorch1.3]  
pillow版本过高导致，安装版本号小于7.0.0即可