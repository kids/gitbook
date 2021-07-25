# contour等高线

数据来源：SRTM、 DLR、ASTER、 GDEM

数据格式，以NASADEM为例：NASADEM\_HGT\_n31e119，表示N31到N32，E119到E120之间的区域。解压后三个文件n31e119.swb、n31e119.num、n31e119.hgt

使用hgt文件绘制类似热力图的pixel高程图：

```python
import numpy as np
import plotly.express as px
with open('./NASADEM_HGT_n31e119/n31e119.hgt', 'rb') as hgt_data:
    elevations = np.fromfile(hgt_data, np.dtype('>i2'))
l=np.power(elevations.shape,0.5)
px.imshow(elevations.reshape(l,l))
```

之后可以叠加同样经纬度范围内的路网图或其他矢量图，更加直观。

