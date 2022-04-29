# CRNN-Pytorch 记录CRNN的学习

CRNN是2015年提出的一种，端对端的，场景文字识别方法，它采用CNN与RNN的结合来进行学习。它相对于其他算法主要有以下两个特点：

1. 端对端训练，直接输入图片给出结果，而不是把多个训练好的模型进行组合来识别
2. 不需要对图片中的文字进行分割就可以进行识别，可以适应任意长度的序列



CRNN具体的网络结构如下：

注意：为了与论文保持一致，本文的宽高结构均用**宽 × 高**来表示，三维张量格式为**宽 × 高 × 通道数**
*其中k表示卷积核大小(kernel_size)，s表示步长(stride)，p表示填充(padding_size)*

|        Type        |        Configurations        |    Output Size    |
| :----------------: | :--------------------------: | :---------------: |
|       Input        |   W × 32 gray-scale image    |    W × 32 × 1     |
|    Convolution     | #maps:64, k:3 × 3, s:1, p:1  |    W × 32 × 64    |
|     MaxPooling     |      Window:2 × 2, s:2       |   W/2 × 16 × 64   |
|    Convolution     | #maps:128, k:3 × 3, s:1, p:1 |  W/2 × 16 × 128   |
|     MaxPooling     |      Window:2 × 2, s:2       |   W/4 × 8 × 128   |
|    Convolution     | #maps:256, k:3 × 3, s:1, p:1 |   W/4 × 8 × 256   |
|    Convolution     | #maps:256, k:3 × 3, s:1, p:1 |   W/4 × 8 × 256   |
|     MaxPooling     |      Window:1 × 2, s:2       |   W/4 × 4 × 256   |
|    Convolution     | #maps:512, k:3 × 3, s:1, p:1 |   W/4 × 4 × 512   |
| BatchNormalization |              -               |   W/4 × 4 × 512   |
|    Convolution     | #maps:512, k:3 × 3, s:1, p:1 |   W/4 × 4 × 512   |
| BatchNormalization |              -               |   W/4 × 4 × 512   |
|     MaxPooling     |      Window:1 × 2, s:2       |   W/4 × 2 × 512   |
|    Convolution     | #maps:512, k:2 × 2, s:1, p:0 |  W/4-1 × 1 × 512  |
|  Map-to-Sequence   |              -               |    W/4-1 × 512    |
| Bidirectional-LSTM |      #hidden units:256       |    W/4-1 × 256    |
| Bidirectional-LSTM |      #hidden units:256       | W/4-1 × label_num |
|   Transcription    |              -               |        str        |



### 卷积

从上表的配置可以看出，卷积层很像VGG-11。不同的地方主要有两个：

1. 增加了批归一化层
2. 池化层的大小从正方形变成了长方形

加入批归一化层可以加快训练。而用高为2宽为1的长方形更容易获取窄长英文字母的特征，这样更容易区分像i和l这样的字母。

参考

- https://github.com/ypwhs/captcha_break
- https://github.com/luoqianlin/deep-learning-demo
- https://github.com/zhaobomin/crnn.pytorch-ocr-train