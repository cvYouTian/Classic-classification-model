# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
)
from matplotlib import pyplot as plt
import numpy as np
from Multi_Model.LeNet5 import Lenet5
import cv2


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
# print(x_train.shape.as_list())
# 查看测试集的图片
# imgs = x_test[:8]
# labs = y_test[:8]
# print(labs)
# h = np.hstack(imgs)
# plt.imshow(h, cmap='gray')
# plt.show()
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
x_train, x_test = tf.image.resize_with_pad(x_train, 32, 32), tf.image.resize_with_pad(x_test, 32, 32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model = Lenet5()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam()

# 训练集的loss与准确率
train_loss = tf.keras.metrics.Mean(name="trian_loss")
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# 验证集的loss与准确率
test_loss = tf.keras.metrics.Mean(name="test_loss")
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

callbacks = [ReduceLROnPlateau(verbose=1),
             EarlyStopping(patience=2, verbose=1),
             ModelCheckpoint('../../checkpoints/{epoch}.tf',
                             verbose=1,
                             save_weights_only=True,
                             period=1)]

history = model.fit(train_ds,
          epochs=10,
          callbacks=callbacks,
          validation_data=test_ds)
