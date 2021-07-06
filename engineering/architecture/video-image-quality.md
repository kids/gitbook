---
description: 视频质量评估
---

# video/image quality

#### 数据源

图像美学数据集aesthetic visual analysis \(AVA\)

CID2013和BID为包含真实噪声的数据集，LIVE、CSIQ、TID2008和TID2013数据集为合成失真

#### 指标

平均主观得分\(Mean opinion score,MOS\)

平均主观得分差\(Diﬀerential mean opinion score,DMOS\)

#### 图片清晰度

import cv2

def getImageVar\(imgPath\):

 image = cv2.imread\(imgPath\);

 img2gray = cv2.cvtColor\(image, cv2.COLOR\_BGR2GRAY\)

 imageVar = cv2.Laplacian\(img2gray, cv2.CV\_64F\).var\(\)

 return imageVar

图片laplacian原理？

越大越清晰，背景虚化badcase

