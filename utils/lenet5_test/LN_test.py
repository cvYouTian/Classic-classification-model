# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from Multi_Model.LeNet5 import Lenet5
import matplotlib.pyplot as plt


# TODO: 下面是lenet5的模型detect实现
# code
# 加载权重
model = Lenet5()
model.load_weights(r'../../checkpoints/8.tf').expect_partial()

imgs = cv2.imread('./numbers/15.jpg')
imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)

imgs = imgs / 255
imgs = tf.expand_dims(imgs, axis=0)
imgs = tf.expand_dims(imgs, axis=-1)
imgs = tf.image.resize(imgs, [32, 32])

imgs = tf.cast(imgs, dtype=tf.float64)

# 加载模型
res = model(imgs)
re = tf.squeeze(res, axis=0)
# tf的排序函数
# re = tf.sort(re)
print(tf.argmax(re))
# tf删除一维
imgs = tf.squeeze(imgs, axis=0)
imgs = tf.squeeze(imgs, axis=-1)

plt.imshow(imgs)
plt.show()


# tf转numpy
# imgs = imgs.numpy()

# 建一个可以调动的窗口
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow("image", imgs)
# cv2.waitKey()
# cv2.destroyAllWindows()


# TODO 笔记:
# opencv得到的图像都是numpy类型的
# 要用tf.cast()进行转化
# tf得到的图形是tf形
# 要用image.numpy()来转换
# 而plt是最猛的，两类数据都可以读取

# 用tf的方式读取一张图片
# imgs = tf.image.decode_image(open('./3.png', 'rb').read(), channels=3)
# plt.imshow(imgs)
# plt.show()
# print(imgs.shape.as_list())
#
# 打印image中的数据类型
# print(imgs.dtype)
#
# tf扩一维
# imgs = tf.expand_dims(imgs, 0)
#
# 打印tf的shape
# print(imgs.shape.as_list())
#
# plt显示
# plt.imshow(imgs)
# plt.show()
#
# 灰度图
# imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
#
# 另一种转rgb的方法
# b, g, r = cv2.split(imgs)
# imgs = cv2.merge([r, g, b])
#
# 这是enumerate（zip（））连用
# for k, (i, l) in enumerate(zip(imgs, labs)):
#      cv2.imwrite(r"C:\Users\lenovo\Desktop\{}{}.jpg".format(l, k), i)

