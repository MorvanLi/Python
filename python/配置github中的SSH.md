 <center>
     <h1>配置github中的SSH key值 实现免输密码</h1>
 </center>

## 1.**设置用户名和邮箱**

```latex
git config --global user.name morvanli
git config --global user.email morvanli@foxmail.com
```



## 2.生成**SSH key**

1. **cd ~/.ssh** 
2. **ssh-keygen -t rsa -C  morvanli@foxmail.com（建议一直回车，不用设置密码）**
3.  **cat id_rsa.pub （复制里面所有的内容）**



## 3. **检查SSH key是否有效**

​    	在git命令行输入：ssh  -T git@github.com; 这里会要求你输入SSH key密码，如果刚才生成SSH key时未输入密码，密码就为空；然后看到信息： **Hi MorvanLi! You've successfully authenticated**；说明配置成功

