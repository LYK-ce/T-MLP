#Presented by KeJi
#Date ： 2026-02-12

"""
数据集下载脚本
支持下载 Cifar-10, Cifar-100, ImageNet, GLUE 数据集
用法: python Dataset_download.py --dataset <数据集名称> --dir <下载路径>
"""

import argparse
import os
import sys


def Parse_Arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='下载机器学习数据集')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['Cifar-10', 'Cifar-100', 'ImageNet', 'GLUE'],
        help='要下载的数据集名称 (Cifar-10, Cifar-100, ImageNet, GLUE)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='数据集下载目标路径'
    )
    return parser.parse_args()


def Create_Directory(dir_path):
    """创建目录（如果不存在）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录: {dir_path}")
    else:
        print(f"目录已存在: {dir_path}")


def Download_Cifar10(save_dir):
    """下载Cifar-10数据集"""
    try:
        import torchvision
        import torchvision.datasets as datasets
        
        dataset_dir = os.path.join(save_dir, 'cifar-10')
        Create_Directory(dataset_dir)
        
        print("开始下载 Cifar-10 数据集...")
        train_dataset = datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            download=True
        )
        test_dataset = datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            download=True
        )
        print(f"Cifar-10 数据集下载完成！保存路径: {dataset_dir}")
        return True
    except Exception as e:
        print(f"下载 Cifar-10 失败: {str(e)}")
        return False


def Download_Cifar100(save_dir):
    """下载Cifar-100数据集"""
    try:
        import torchvision
        import torchvision.datasets as datasets
        
        dataset_dir = os.path.join(save_dir, 'cifar-100')
        Create_Directory(dataset_dir)
        
        print("开始下载 Cifar-100 数据集...")
        train_dataset = datasets.CIFAR100(
            root=dataset_dir,
            train=True,
            download=True
        )
        test_dataset = datasets.CIFAR100(
            root=dataset_dir,
            train=False,
            download=True
        )
        print(f"Cifar-100 数据集下载完成！保存路径: {dataset_dir}")
        return True
    except Exception as e:
        print(f"下载 Cifar-100 失败: {str(e)}")
        return False


def Download_ImageNet(save_dir):
    """ImageNet数据集需要手动下载"""
    dataset_dir = os.path.join(save_dir, 'imagenet')
    Create_Directory(dataset_dir)
    
    print("=" * 60)
    print("ImageNet 数据集需要手动下载")
    print("=" * 60)
    print("请按以下步骤操作:")
    print("1. 访问 https://image-net.org/download.php")
    print("2. 注册账号并申请下载权限")
    print("3. 下载 ILSVRC2012 数据集:")
    print("   - Training images (138GB)")
    print("   - Validation images (6.3GB)")
    print("   - Test images (13GB)")
    print(f"4. 将下载的文件解压到: {dataset_dir}")
    print("=" * 60)
    
    readme_path = os.path.join(dataset_dir, 'README.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("ImageNet 数据集下载说明\\n")
        f.write("=" * 60 + "\\n\\n")
        f.write("1. 访问 https://image-net.org/download.php\\n")
        f.write("2. 注册账号并申请下载权限\\n")
        f.write("3. 下载 ILSVRC2012 数据集\\n")
        f.write("4. 解压到本目录\\n")
    
    print(f"已在 {dataset_dir} 创建 README.txt 文件")
    return True


def Download_GLUE(save_dir):
    """下载GLUE文本数据集"""
    try:
        from datasets import load_dataset
        
        dataset_dir = os.path.join(save_dir, 'glue')
        Create_Directory(dataset_dir)
        
        # GLUE包含多个子任务
        glue_tasks = [
            'cola', 'sst2', 'mrpc', 'qqp', 'stsb',
            'mnli', 'qnli', 'rte', 'wnli'
        ]
        
        print("开始下载 GLUE 数据集...")
        for task in glue_tasks:
            print(f"  下载 {task}...")
            try:
                dataset = load_dataset('glue', task)
                task_dir = os.path.join(dataset_dir, task)
                Create_Directory(task_dir)
                
                # 保存为JSON格式
                for split in dataset.keys():
                    output_file = os.path.join(task_dir, f'{split}.json')
                    dataset[split].to_json(output_file)
                    print(f"    {split}: {len(dataset[split])} 条记录")
            except Exception as e:
                print(f"    {task} 下载失败: {str(e)}")
        
        print(f"GLUE 数据集下载完成！保存路径: {dataset_dir}")
        return True
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("请运行: pip install datasets")
        return False
    except Exception as e:
        print(f"下载 GLUE 失败: {str(e)}")
        return False


def Download_Dataset(dataset_name, save_dir):
    """根据数据集名称调用对应的下载函数"""
    download_functions = {
        'Cifar-10': Download_Cifar10,
        'Cifar-100': Download_Cifar100,
        'ImageNet': Download_ImageNet,
        'GLUE': Download_GLUE
    }
    
    if dataset_name not in download_functions:
        print(f"错误: 不支持的数据集 '{dataset_name}'")
        return False
    
    return download_functions[dataset_name](save_dir)


def Check_Dependencies(dataset_name):
    """检查数据集依赖的库是否已安装"""
    missing_deps = []
    
    if dataset_name in ['Cifar-10', 'Cifar-100']:
        try:
            import torchvision
        except ImportError:
            missing_deps.append('torchvision')
    
    if dataset_name == 'GLUE':
        try:
            import datasets
        except ImportError:
            missing_deps.append('datasets')
    
    if missing_deps:
        print(f"错误: 缺少必要的依赖库: {', '.join(missing_deps)}")
        print(f"请运行: pip install {' '.join(missing_deps)}")
        return False
    
    return True


def Main():
    """主函数"""
    args = Parse_Arguments()
    
    # 检查依赖
    if not Check_Dependencies(args.dataset):
        sys.exit(1)
    
    # 创建目标目录
    Create_Directory(args.dir)
    
    # 下载数据集
    success = Download_Dataset(args.dataset, args.dir)
    
    if success:
        print(f"\\n数据集 {args.dataset} 准备完成！")
        sys.exit(0)
    else:
        print(f"\\n数据集 {args.dataset} 准备失败！")
        sys.exit(1)


if __name__ == '__main__':
    Main()
