# T-MLP 工作记录

## 任务状态
1. [x] 分析方案可行性并补充评价 - 2026-02-12
2. [x] 数据准备脚本实现 - 2026-02-12 完成
   - 依赖: 无
   - 目标: Tool/Dataset_download.py 支持 Cifar-10/100, ImageNet, GLUE
   - 文件: Tool/Dataset_download.py
   - 用法: python Dataset_download.py --dataset <名称> --dir <路径>
   - 实现功能:
     - Cifar-10/100: 使用urllib从Toronto官网下载tar.gz并解压
     - ImageNet: 生成下载说明(需手动下载)
     - GLUE: 使用urllib从Facebook官方链接下载所有10个子任务
       - CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI, AX
       - MRPC使用带User-Agent头的请求解决403错误
   - 特点: 纯标准库实现，无需安装额外依赖
3. [x] 模型实现 - 2026-02-12 完成
   - 依赖: 无
   - 目标: Model/VGG_16.py 和 Model/Deit_Small.py
   - 文件: Model/VGG_16.py, Model/Deit_Small.py
   - VGG_16.py:
     - 类: VGG(nn.Module)
     - 属性: num_classes, input_size, weight_path
     - 方法: __init__(num_classes=1000, input_size=224, weight_path=None), forward(x), Save_Model_Weight(weight_path)
     - 结构: 5个卷积块 + 动态计算全连接层输入维度 + 3层全连接分类器
     - 动态计算: flattened_size = 512 * (input_size // 32)^2
     - 适配: 支持CIFAR-100(32x32→512)和ImageNet(224x224→25088)等不同输入尺寸
   - Deit_Small.py:
     - 类: Deit(nn.Module), PatchEmbed, Attention, Mlp, Block
     - 属性: num_classes, input_size, weight_path
     - 方法: __init__(num_classes=1000, input_size=224, weight_path=None, ...), forward(x), Save_Model_Weight(weight_path)
     - 结构: Patch嵌入 + 位置编码 + 12个Transformer块 + LayerNorm + Head
     - 配置: embed_dim=384, depth=12, num_heads=6, mlp_ratio=4
4. [x] 模型调整 - 2026-02-12 完成
   - 依赖: 任务3
   - 目标: 根据新Design.md创建4个模型文件
   - Cifar-10: num_classes=10, input_size=32
   - Cifar-100: num_classes=100, input_size=32
   - VGG_Cifar10: 512*1*1=512, 保存权重vgg16_cifar10
   - VGG_Cifar100: 512*1*1=512, 保存权重vgg16_cifar100
   - Deit_Cifar10: patch_size=4, num_patches=64, 保存权重deit_cifar10
   - Deit_Cifar100: patch_size=4, num_patches=64, 保存权重deit_cifar100
   - 文件:
     - Model/VGG_16_Cifar10.py (Class VGG_Cifar10)
     - Model/VGG_16_Cifar100.py (Class VGG_Cifar100)
     - Model/Deit_Small_Cifar10.py (Class Deit_Cifar10)
     - Model/Deit_Small_Cifar100.py (Class Deit_Cifar100)
5. [x] 训练脚本实现 - 2026-02-12 完成
   - 依赖: 任务4(模型调整)
   - 目标: Src/train.py, train.sh
   - 开始时间: 2026-02-12T09:03
   - 结束时间: 2026-02-12T09:05
   - 参数: --model, --dataset, --path, --epoch, --lr, --batch_size, --weight_decay, --device, --num_workers
   - 功能: 每10epoch打印准确率, 保存best checkpoint, 最终保存best权重
   - Src/train.py:
     - 支持VGG16/Deit + Cifar-10/100组合
     - Get_Model(): 动态导入模型类
     - Get_Dataset(): CIFAR数据加载与预处理
     - Evaluate_Accuracy(): 测试集准确率评估
     - Train_Model(): 训练循环, SGD优化器, CosineAnnealingLR调度
     - 保存best_checkpoint.pth, 最终调用Save_Model_Weight()
     - 数据路径: /home/dataset-local/lyk/Data
   - train.sh:
     - Linux批处理脚本
     - 顺序训练4个任务: VGG16_C10, VGG16_C100, Deit_C10, Deit_C100
     - 默认参数: epoch=200, lr=0.1, batch=128
6. [x] ImageNet模型实现 - 2026-02-12 完成
   - 依赖: 无
   - 目标: Model/VGG_16_ImageNet.py, Model/Deit_Small_ImageNet.py
   - 开始时间: 2026-02-12T12:11
   - 结束时间: 2026-02-12T12:13
   - ImageNet配置: num_classes=1000, input_size=224
   - VGG_ImageNet: flattened_size=512*7*7=25088, 保存权重vgg16_imagenet
   - Deit_ImageNet: patch_size=16, num_patches=196, 保存权重deit_imagenet
   - 文件:
     - Model/VGG_16_ImageNet.py (Class VGG_ImageNet)
     - Model/Deit_Small_ImageNet.py (Class Deit_ImageNet)
   - Src/train.py更新:
     - Get_Model()支持ImageNet模型导入
     - Get_Dataset()支持ImageNet数据加载(ImageNet的train/val split)
     --dataset参数添加ImageNet选项
   - train.sh更新:
     - 从4个任务扩展到6个任务
     - 添加VGG16_ImageNet训练(epoch=90, lr=0.01, batch=256)
     - 添加Deit_ImageNet训练(epoch=300, lr=0.001, batch=256)
