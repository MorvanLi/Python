 <center>
     <h1>pytorch常用操作日常记录</h1>
 </center>

##  1 、 如何求tensor的均值和方差

1. [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html?highlight=torch%20mean#torch.mean)
2. [torch.val](https://pytorch.org/docs/stable/generated/torch.std.html?highlight=torch%20std#torch.std)
3. [torch.var_mean](https://pytorch.org/docs/stable/generated/torch.var_mean.html?highlight=torch%20var_mean#torch.var_mean)

## 2、如何交换维度顺序

1. [torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html?highlight=transpose#torch.transpose)
2. [torch.permute](https://pytorch.org/docs/stable/generated/torch.permute.html#torch.permute)

区别在于，transpose一次只能完成两个维度的转换，permute可以同时完成多个维度转换。



## 3、如何计算tensor的形状

1. tensor.shape

2. tensor.size()

   

## 4、如何拼接和切割tensor

1. [tensor.cat](https://pytorch.org/docs/stable/generated/torch.cat.html?highlight=cat#torch.cat)
2. [tensor.stack](https://pytorch.org/docs/stable/generated/torch.stack.html?highlight=torch%20stack#torch.stack)
3. [tensor.split](https://pytorch.org/docs/stable/generated/torch.split.html#torch.split)
4. [tensor.chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk)



