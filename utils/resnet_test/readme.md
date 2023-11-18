# Note

- b站视频[ ResNet网络结构，BN以及迁移学习详解](https://www.bilibili.com/video/BV1T7411T7wa?spm_id_from=333.999.0.0)
- 训练的文件train
    * 本次训练都用了有使用迁移学习的方式进行训练
        + 在进行迁移学习训练时一定要进行与权重匹配的预处理方法，所以用生成器构建数据集时用到了pre_function函数
    * train.py用的是从零开始的训练
    * train2.py用的是预训练的方法（底层api实现）
    * trainGPU.py
- 关于模型
    * ResNet.py主要用的是函数api实现
    * ResNet_2.py主要用的是函数api实现
    * ResNet_3.py主要用的子类实现

- 关于预训练loadweights.py(pytorch版)
    * [torch.load_state_dict()函数的用法总结](https://blog.csdn.net/ChaoMartin/article/details/118686268)
        + 预训练时，如果令参数为strict=True就是说预训练的参数层key必须和自己的模型一样， 如果增加层数的话就会报错，此时可以令strict=false
        就可以解决，如果改了某层的结构，key并没有变化但是参数不在匹配还是会报错、

    
    

# 小结
- 引入组卷积来减少卷积的参数量
    * ![groupConv是如何减少参数的](./groupConv.png)
    * DW卷积的使用
        + [Depthwise卷积与Pointwise卷积](https://blog.csdn.net/tintinetmilou/article/details/81607721)
- resnet提出了residual模块让网络突破了1000层
    * ![resdu](./res.png)
- 用Batch Normalization替代了dropout训练
    * batchNormalization的目的是得到的一批图像的featuremap的同个channel满足均值为0， 方差为1的分布规律
        + [Batch Normalization详解以及pytorch实验](https://blog.csdn.net/qq_37541097/article/details/104434557)
    

