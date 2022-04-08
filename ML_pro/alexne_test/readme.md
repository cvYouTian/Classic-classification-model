# Note

- Relu里的（inplace=True）的意思是进行原地操作，例如x=x+5，对x就是一个原地操作，y=x+5,x=y，完成了与x=x+5同样的功能但是不是原地操作，上面LeakyReLU中的inplace=True的含义是一样的，是对于Conv2d这样的上层网络传递下来的tensor直接进行修改，好处就是可以节省运算内存，不用多储存变量y。
- [type() 与 isinstance()区别](https://www.runoob.com/python/python-func-isinstance.html)
- 