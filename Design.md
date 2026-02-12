# T-MLP 设计文档

## 数据准备
我们将会准备Cifar-10，Cifar-100，ImageNet图像数据集，以及GLUE文本数据集。

### 数据集下载脚本
Tool/Dataset_download.py
使用方法 python Tool/Dataset_download.py --dataset Cifar-10 --dir /Path

此数据集将会下载对应数据集到对应的Path当中。