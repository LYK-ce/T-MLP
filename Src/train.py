'''
#Presented by KeJi
#Date ： 2026-02-12
'''

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def Get_Model(model_name, dataset_name):
    if model_name == 'VGG16':
        if dataset_name == 'Cifar-10':
            from Model.VGG_16_Cifar10 import VGG_Cifar10
            return VGG_Cifar10()
        elif dataset_name == 'Cifar-100':
            from Model.VGG_16_Cifar100 import VGG_Cifar100
            return VGG_Cifar100()
    elif model_name == 'Deit':
        if dataset_name == 'Cifar-10':
            from Model.Deit_Small_Cifar10 import Deit_Cifar10
            return Deit_Cifar10()
        elif dataset_name == 'Cifar-100':
            from Model.Deit_Small_Cifar100 import Deit_Cifar100
            return Deit_Cifar100()
    raise ValueError(f"Unsupported model {model_name} or dataset {dataset_name}")


def Get_Dataset(dataset_name, base_data_dir):
    if dataset_name == 'Cifar-10':
        data_dir = os.path.join(base_data_dir, 'cifar-10')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
        num_classes = 10
    elif dataset_name == 'Cifar-100':
        data_dir = os.path.join(base_data_dir, 'cifar-100')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")
    return train_set, test_set, num_classes


def Evaluate_Accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def Train_Model(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    data_dir = '/home/dataset-local/lyk/Data'
    train_set, test_set, num_classes = Get_Dataset(args.dataset, data_dir)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = Get_Model(args.model, args.dataset)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    
    best_accuracy = 0.0
    best_epoch = 0
    
    os.makedirs(args.path, exist_ok=True)
    
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        train_accuracy = 100.0 * train_correct / train_total
        test_accuracy = Evaluate_Accuracy(model, test_loader, device)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            checkpoint_path = os.path.join(args.path, 'best_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }, checkpoint_path)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{args.epoch}] Train Acc: {train_accuracy:.2f}% Test Acc: {test_accuracy:.2f}% Best Acc: {best_accuracy:.2f}%')
    
    checkpoint_path = os.path.join(args.path, 'best_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded best checkpoint from epoch {checkpoint["epoch"]+1} with accuracy {checkpoint["best_accuracy"]:.2f}%')
    
    model.Save_Model_Weight(args.path)
    print(f'Training completed. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch+1}')
    print(f'Model weights saved to {args.path}')


def main():
    parser = argparse.ArgumentParser(description='Train model on dataset')
    parser.add_argument('--model', type=str, required=True, choices=['VGG16', 'Deit'], help='Model name')
    parser.add_argument('--dataset', type=str, required=True, choices=['Cifar-10', 'Cifar-100'], help='Dataset name')
    parser.add_argument('--path', type=str, default='ModelWeights', help='Path to save model weights')
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    Train_Model(args)


if __name__ == '__main__':
    main()
