# Note

# 小结

- mobileNetv1网络引入了Depthwise Convolution(深层卷积)
    - dw卷积+pw卷积（pointwise， 其实就是尺寸为1*1的卷积）=深度可分离卷积
    - 节省的参数量
    ![深度分离卷积减少的参数量](./深度可分离卷积.png)
- v1增加了超参数a（depthwise），b