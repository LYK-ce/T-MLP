# T-MLP 设计文档

## 核心思想

### 泰勒展开近似
多层感知机（Multi-Layer Perceptron, MLP）可形式化表示为复合函数 $F(X) = f_L \circ f_{L-1} \circ \dots \circ f_1(X)$，其中每一层 $f_i(X) = \sigma(W_i X + b_i)$ 包含线性变换与激活函数。传统前向传播需执行多次矩阵乘法与非线性运算，计算开销显著。

本方案采用泰勒展开（Taylor Expansion）对MLP进行局部线性近似。对于展开点 $X_0$，其一阶泰勒展开式为：

$$F(X) \approx F(X_0) + J(X_0)(X - X_0)$$

其中 $J(X_0)$ 为Jacobian矩阵。若采用二阶展开，则引入Hessian矩阵 $H(X_0)$：

$$F(X) \approx F(X_0) + J(X_0)(X - X_0) + \frac{1}{2}(X - X_0)^T H(X_0)(X - X_0)$$

通过预计算展开点处的函数值 $F(X_0)$、Jacobian矩阵 $J(X_0)$ 及Hessian矩阵 $H(X_0)$，推理阶段仅需执行一次矩阵乘加运算（一阶）或附加二次项计算（二阶），从而显著降低计算复杂度。

### K-means聚类展开点选取

展开点的选取对近似精度具有决定性影响：输入样本距展开点越近，泰勒近似的截断误差越小。为此，本方案采用K-means聚类算法对训练集中MLP的输入特征进行聚类分析，以聚类中心作为展开点集合 $\{X_0^{(1)}, X_0^{(2)}, \dots, X_0^{(k)}\}$，其中 $k$ 为预设的展开点数量。

推理阶段，对于输入 $X$，首先计算其与各展开点的欧氏距离，选取最近展开点进行泰勒展开：

$$c = \arg\min_{i} \|X - X_0^{(i)}\|_2$$

### 复杂度分析

#### 计算复杂度

| 方案 | 计算复杂度 | 说明 |
|-----|-----------|------|
| 原始MLP | $O(L \cdot d^2)$ | $L$ 为层数，$d$ 为特征维度 |
| T-MLP一阶 | $O(k \cdot d + d^2)$ | 含K-means最近邻查找与Jacobian矩阵乘法 |
| T-MLP二阶(对角近似) | $O(k \cdot d + d^2)$ | 对角Hessian近似，额外计算开销为 $O(d)$ |

**分析**：当展开点数量 $k \ll L \cdot d$ 时，T-MLP可实现显著加速。初步实验表明，在资源受限设备上可获得两个数量级的推理加速。

#### 空间复杂度

以DeiT-Base模型（$d=768$）为例，各方案存储需求对比如下：

| 方案 | 存储维度 | 单展开点 | k=100总存储(12层) |
|-----|---------|---------|------------------|
| 原始MLP权重 | - | - | ~36 MB |
| 一阶(Jacobian) | $768 \times 768$ | 2.36 MB | **2.83 GB** |
| 二阶(完整Hessian) | $768 \times 768 \times 768$ | 1.81 GB | **2.17 TB** |
| 二阶(对角Hessian) | $768 \times 768$ | 2.36 MB | **2.83 GB** |

**关键结论**：
1. 完整Hessian张量的存储复杂度为 $O(d^3)$，在实际应用中不可行
2. 对角Hessian近似将存储复杂度降至 $O(d^2)$，与一阶展开同量级
3. 一阶展开存储开销约为原始权重的80倍，需通过优化展开点数量 $k$ 或采用低秩近似等方法进一步压缩

## 数据准备
我们将会准备Cifar-10，Cifar-100，ImageNet图像数据集，以及GLUE文本数据集。

### 数据集下载脚本
Tool/Dataset_download.py
使用方法 python Tool/Dataset_download.py --dataset Cifar-10 --dir /Path

此数据集将会下载对应数据集到对应的Path当中。

## 模型
为了实现我们的idea，即使用泰勒展开+kmeans聚类的方法代替当前计算量极其大的MLP，我们需要对原来的代码进行一定的修改，首先就是可以保存中间结果，然后是可以选择使用T-MLP的方案代替原来的MLP部分，最后，我们还需要在其中增加一些记录时间的代码，用来计算时间长短。

### 实验模型
我们的实验包括以下模型：
- VGG-16
- Deit-Small
之后可能还会增加一些其他模型

方便起见，我们为每一个数据集都准备一个脚本文件
### 模型文件
Model/VGG_16_Cifar10.py

Class VGG_Cifar10
属性：
- num_calsses
- input_size

- weight_path

方法
- 初始化方法
- forward方法
- Save_Model_Weight(weight_path)
    将当前模型权重"vgg16_cifar10"保存到weight_path目录下

Model/VGG_16_Cifar100.py

Class VGG_Cifar100
属性：
- num_calsses
- input_size

- weight_path

方法
- 初始化方法
- forward方法
- Save_Model_Weight(weight_path)
    将当前模型权重"vgg16_cifar100"保存到weight_path目录下


Model/Deit_Small_Cifar10.py

Class Deit_Cifar10
属性：
- num_calsses
- input_size

- weight_path

方法
- 初始化方法
- forward方法
- Save_Model_Weight(weight_path)
    将当前模型权重"deit_cifar10"保存到weight_path目录下

Model/Deit_Small_Cifar100.py

Class Deit_Cifar100
属性：
- num_calsses
- input_size

- weight_path

方法
- 初始化方法
- forward方法
- Save_Model_Weight(weight_path)
    将当前模型权重"deit_cifar100"保存到weight_path目录下



## 训练
我们需要先提前准备好训练好的模型进行实验。

## 训练脚本
Src/train.py 
参数：
-- model    指定的模型
-- dataset  指定的数据集
-- path     模型权重保存的路径，默认为ModelWeights
-- epoch    模型训练轮数
-- lr       学习率
-- batch_size   batch size大小
-- weight_decay 
-- device
-- num_workers  默认为4

训练过程中每10个epoch打印一次准确率，另外，将会保存一个best checkpoint，最终模型保存的权重就是这个bestcheckpoint。

train.sh
此脚本将运行在linux环境下，因此严格按照linux的要求来。
此脚本将会分别在cifar 10 cifar 100上训练VGG16和Deit
