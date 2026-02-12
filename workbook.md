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
