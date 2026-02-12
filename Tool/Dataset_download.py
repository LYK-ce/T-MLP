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
import urllib.request
import tarfile
import zipfile
import json


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


def Download_File(url, save_path, description=""):
    """使用urllib下载文件"""
    try:
        print(f"下载 {description or url} ...")
        
        def Progress_Hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                percent = min(percent, 100)
                print(f"\r进度: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, save_path, Progress_Hook)
        print()  # 换行
        return True
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        return False


def Extract_Archive(archive_path, extract_dir):
    """解压压缩文件"""
    try:
        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        print(f"解压完成: {extract_dir}")
        return True
    except Exception as e:
        print(f"解压失败: {str(e)}")
        return False


def Download_Cifar10(save_dir):
    """下载Cifar-10数据集"""
    dataset_dir = os.path.join(save_dir, 'cifar-10')
    Create_Directory(dataset_dir)
    
    # Cifar-10 官方下载链接
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    archive_path = os.path.join(dataset_dir, 'cifar-10-python.tar.gz')
    
    if os.path.exists(archive_path.replace('.tar.gz', '')):
        print("Cifar-10 已存在，跳过下载")
        return True
    
    if Download_File(url, archive_path, "Cifar-10"):
        if Extract_Archive(archive_path, dataset_dir):
            # 删除压缩包
            os.remove(archive_path)
            print(f"Cifar-10 数据集下载完成！保存路径: {dataset_dir}")
            return True
    return False


def Download_Cifar100(save_dir):
    """下载Cifar-100数据集"""
    dataset_dir = os.path.join(save_dir, 'cifar-100')
    Create_Directory(dataset_dir)
    
    # Cifar-100 官方下载链接
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    archive_path = os.path.join(dataset_dir, 'cifar-100-python.tar.gz')
    
    if os.path.exists(archive_path.replace('.tar.gz', '')):
        print("Cifar-100 已存在，跳过下载")
        return True
    
    if Download_File(url, archive_path, "Cifar-100"):
        if Extract_Archive(archive_path, dataset_dir):
            os.remove(archive_path)
            print(f"Cifar-100 数据集下载完成！保存路径: {dataset_dir}")
            return True
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
        f.write("ImageNet 数据集下载说明\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. 访问 https://image-net.org/download.php\n")
        f.write("2. 注册账号并申请下载权限\n")
        f.write("3. 下载 ILSVRC2012 数据集\n")
        f.write("4. 解压到本目录\n")
    
    print(f"已在 {dataset_dir} 创建 README.txt 文件")
    return True


def Download_File_With_Headers(url, save_path, description=""):
    """使用urllib下载文件（带请求头）"""
    try:
        print(f"下载 {description or url} ...")
        
        # 创建请求并添加User-Agent头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        print(f"\r进度: {percent}%", end='', flush=True)
        
        print()  # 换行
        return True
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        return False


def Download_GLUE(save_dir):
    """下载GLUE文本数据集 - 使用urllib直接从官方下载"""
    dataset_dir = os.path.join(save_dir, 'glue')
    Create_Directory(dataset_dir)
    
    # GLUE 子任务列表（包含AX诊断任务）
    glue_tasks = ['CoLA', 'SST-2', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'QNLI', 'RTE', 'WNLI', 'AX']
    
    # Hugging Face datasets 原始文件基础URL
    base_url = "https://dl.fbaipublicfiles.com/glue/data"
    
    # 任务到下载链接的映射
    # MRPC使用Hugging Face镜像（Facebook官方403限制）
    task_urls = {
        'CoLA': f"{base_url}/CoLA.zip",
        'SST-2': f"{base_url}/SST-2.zip",
        'MRPC': "https://huggingface.co/datasets/glue/resolve/main/data/MRPC.zip",
        'QQP': f"{base_url}/QQP-clean.zip",
        'STS-B': f"{base_url}/STS-B.zip",
        'MNLI': f"{base_url}/MNLI.zip",
        'QNLI': f"{base_url}/QNLIv2.zip",
        'RTE': f"{base_url}/RTE.zip",
        'WNLI': f"{base_url}/WNLI.zip",
        'AX': f"{base_url}/AX.tsv"
    }
    
    print("开始下载 GLUE 数据集...")
    success_count = 0
    
    for task in glue_tasks:
        task_dir = os.path.join(dataset_dir, task)
        Create_Directory(task_dir)
        
        url = task_urls.get(task)
        if not url:
            continue
        
        # 检查是否已存在
        if os.listdir(task_dir):
            print(f"  {task}: 已存在，跳过")
            success_count += 1
            continue
        
        print(f"  下载 {task}...")
        
        if url.endswith('.tsv'):
            # 直接下载tsv文件（使用带headers的版本）
            save_path = os.path.join(task_dir, 'test.tsv')
            if Download_File_With_Headers(url, save_path, f"{task} test"):
                success_count += 1
        else:
            # 下载并解压zip文件
            zip_path = os.path.join(dataset_dir, f'{task}.zip')
            # MRPC需要特殊处理403错误，使用带headers的版本
            if task == 'MRPC':
                success = Download_File_With_Headers(url, zip_path, task)
            else:
                success = Download_File(url, zip_path, task)
            
            if success and Extract_Archive(zip_path, dataset_dir):
                os.remove(zip_path)
                success_count += 1
    
    print(f"\nGLUE 数据集下载完成！({success_count}/{len(glue_tasks)} 个任务)")
    print(f"保存路径: {dataset_dir}")
    return success_count > 0


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


def Main():
    """主函数"""
    args = Parse_Arguments()
    
    # 创建目标目录
    Create_Directory(args.dir)
    
    # 下载数据集
    success = Download_Dataset(args.dataset, args.dir)
    
    if success:
        print(f"\n数据集 {args.dataset} 准备完成！")
        sys.exit(0)
    else:
        print(f"\n数据集 {args.dataset} 准备失败！")
        sys.exit(1)


if __name__ == '__main__':
    Main()
