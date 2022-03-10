## Receptive field

**The convolutional neural network, a decision of an element in the output layer corresponds to the input layer of the size of the region, known as receptive field (receptive field). Popular explanation is that the output feature on the map of a unit corresponding to the input layer on the size of the region.**

**在卷积神经网络中，决定某一层输出结果所对应的输入层的区域大小，被称作为感受野（receptive field）。通俗的解释是，输出feature map上的一个单元对应输入层的区域大小。**



## Receptive field  formula

$ F(i)=(F(i+1) -1 )*stride + Ksize $

- $F(i)$为第$i$层的感受野
- $stride$为第$i$层的步距
- $Ksize$为卷积核的尺寸



***

**可以通过堆叠两个3x3的卷积核替代5x5的卷积核，堆叠三个3x3的卷积核替代7x7的卷积核**。

1. **减少模型的参数**
2. **增加模型的非线性能力**

**使用7x7的卷积核所需参数，与堆叠三个3x3卷积核所需参数（假设输入输出的channel都是C，并且忽略bias）**

$7*7*C*C = 49C $

$3*3*C*C +3*3*C*C +3*3*C*C =27C$