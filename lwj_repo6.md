# 第六次作业

廖沩健

自动化65

2160504124

提交日期:  2019年4月1日

摘要:

这次作业聚焦于图像的噪声添加与恢复。其中，选择的高斯噪声的均值为0,标准差为25，椒盐噪声噪声对应的脉冲概率为0.03。实现的恢复滤波器有代数平均滤波器，几何平均滤波器，调和平均滤波器，逆谐波滤波器，中值滤波器，最大最小滤波器，midpoint滤波器，alpha trimmed mean滤波器，自适应滤波器，自适应中值滤波器．其中所有的算法除快速傅里叶变换外都是用python从头实现的，使用到的库主要有numpy,cv2等。大部分参考的都是课本与ppt的内容。对于所有处理结果的讨论，都展示在下面的对应的部分。


## 添加高斯噪声后的图像(均值:0, 标准差:25)
![](https://raw.githubusercontent.com/mutewall/homework_img/master/gaussian_0_25.bmp)

当高斯噪声的均值为0，随着方差增加，图像噪声越来越严重；当方差不变，均值的变化会影响到整个图像的灰度值，使整个图像变亮。

## 添加椒盐噪声的图像(脉冲概率：0.03)
![](https://raw.githubusercontent.com/mutewall/homework_img/master/sp0.03.bmp)

## 代数平均滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/am_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/am_sp.bmp)

## 几何平均滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/gm_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/gm_sp.bmp)

## 调和平均滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/hm_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/hm_sp.bmp)

## 逆谐波平均滤波(Q=-0.2)
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_sp.bmp)

## 逆谐波平均滤波(Q=-0.6)
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_g_-0.6.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_sp_-0.6.bmp)

## 逆谐波平均滤波(Q=0.6)
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_g_0.6.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_sp_0.6.bmp)

## 逆谐波平均滤波(Q=0.2)
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_g_0.2.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ihm_sp_0.2.bmp)

## 中值滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/med_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/med_sp.bmp)

## 最大值滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/max_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/max_sp.bmp)

## 最小值滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/min_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/min_sp.bmp)

## midpoint滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/mid_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/mid_sp.bmp)

## alpha_trimmed_mean滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/alpha_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/alpha_sp.bmp)

## 自适应滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ada_g.bmp)

## 自适应中值滤波
### 高斯噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ada_med_g.bmp)
### 椒盐噪声恢复效果
![](https://raw.githubusercontent.com/mutewall/homework_img/master/ada_med_sp.bmp)

由以上结果可知，均值滤波器系列对高斯噪声恢复的效果不错，但一定程度上模糊了图像，而对于椒盐噪声的恢复就比较糟糕，尤其是几何均值，调和均值，加强了＂椒＂，从而遮挡住了原来的图像；顺序统计量滤波器系列对于高斯噪声的恢复的效果不好也不差，对于椒盐噪声的恢复效果两级分化严重，中值滤波器恢复效果非常好，但最大最小滤波器的效果就很不好；自适应滤波对二者的效果都不错；由于题目要求，这里多展示了逆谐波的四种情况，分别是Q等于0.6,0.2,-0.2,-0.6，当Q大于零时，呈现出均值滤波和最小值滤波的叠加效果，且随着绝对值增大，这种效果越明显；当Q小于零时，呈现均值滤波和最大值滤波的叠加效果，且随着绝对值增大，这种效果越明显。
## 模糊滤波
![](https://raw.githubusercontent.com/mutewall/homework_img/master/blur.bmp)
这里设置的参数为:a=0.01, b=0.01, T=1

## 模糊后添加高斯噪声
![](https://raw.githubusercontent.com/mutewall/homework_img/master/blur_gaussian.bmp)
高斯噪声的均值为0，标准差为10。

## Weiner滤波恢复
![](https://raw.githubusercontent.com/mutewall/homework_img/master/restore_blur_gaussian.bmp)

## 约束最小二乘滤波
![](https://raw.githubusercontent.com/mutewall/homework_img/master/restore_blur_gaussian_cls.bmp)

这里展示的维纳滤波和约束最小二乘效果都较为一般．可能是参数设置或者代码实现的原因。二者对于运动模糊的去除的效果都不太好，并且还不合理地增大了图像的亮度，但是对于高斯噪声的祛除却是效果显著。

## 维纳滤波器的推导
由于公式的编辑不是很方便，因此我将在纸上推导的过程拍下，用图片的形式展示。

![](https://raw.githubusercontent.com/mutewall/homework_img/master/1775741539.jpg)

![](https://raw.githubusercontent.com/mutewall/homework_img/master/236463266.jpg)