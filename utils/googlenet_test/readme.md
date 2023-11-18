# Note

- [GoogLeNet系列解读](https://blog.csdn.net/shuzfan/article/details/50738394?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164294285416781685335486%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164294285416781685335486&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-50738394.pc_search_result_cache&utm_term=Googlenet&spm=1018.2226.3001.4187)
- [稀疏连接](https://blog.csdn.net/qq_40234695/article/details/88708756?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1.pc_relevant_default&utm_relevant_index=2)
    * 我的理解是卷积就是稀疏链接，1\*1的卷积核相当把图片打平的全连接（矩阵计算）
- [GoogLenet中子类定义的方法](https://blog.csdn.net/weixin_44560088/article/details/112120979)
- [Python中\*args和\**kwargs的区别](https://www.cnblogs.com/yunguoxiaoqiao/p/7626992.html)
- Googlenet我在Gpu上训练了100个epoch，在第80个epoch收敛，验证准确率达到80，训练集的损失导0.13左右，没有出现过拟合，并行结构牛！！
- 这个predict.用Opencv和matplotlab实现： 
    * 图片尺寸初始化（这里是224）
    * 加载进来一张测试的图片
    * 对其进行归一化处理和扩维（batch_size)维
- padding的计算方法：
    * padding==“valid”时：
        + out = $\frac{s-f+p}{stride}$
    * padding == "same“时：
        + strides==2， out = $\frac{s}{stride}$
        + strides==1， out = in
- 对于不同的train文件按进行说明
    * trainGPU.py中有GPU设备的选择
    * train2.py中有进度条在循环中的用法
    * train.py使用了keras高级api进行搭建、train2.py和trainGPU.py中使用了底层搭建训练，在数据集方面train1和train2都还是使用了keras提
    供的图片生成器，而trainGPU.py用了自己构键数据集的方式
    * 运行train.py（100个epoch的.h5）第74个收敛：
        + 30ms/step - loss: 0.1701 - accuracy: 0.9442 - val_loss: 0.5930 - val_accuracy: 0.7917
    * 运行trainGPU.py(用的GoogleNet.py, 100个epoch的.ckpt)
        + Epoch 100, loss: 0.04066746309399605, Accuracy: 99.45388793945312, val Loss :1.2483785152435303,val Accuracy 81.25
- model有两个一个是BN的一个是无BN的

# 小结

- 1.引入1\*1的卷积核来降维，减少计算参数
    * ![1*1卷积核的作用](./11卷积核作用.png)
- 2.goolenet在当时的分类比赛中拿到了第一名，当时分类的第二名是vggnet，googlenet首次引用了并行的inception模块，和两个辅助输出。首次应用了
并行结构，相对与串行效果很好。