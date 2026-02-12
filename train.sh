#!/bin/bash
#Presented by KeJi
#Date ： 2026-02-12

# T-MLP Training Script for Linux
# This script trains VGG16 and Deit models on Cifar-10, Cifar-100, and ImageNet datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/Src/train.py"
WEIGHT_DIR="$SCRIPT_DIR/ModelWeights"

echo "========================================"
echo "T-MLP Training Pipeline"
echo "========================================"

# Create ModelWeights directory if not exists
mkdir -p "$WEIGHT_DIR"

# Train VGG16 on Cifar-10
echo ""
echo "[1/6] Training VGG16 on Cifar-10..."
python "$TRAIN_SCRIPT" \
    --model VGG16 \
    --dataset Cifar-10 \
    --path "$WEIGHT_DIR/VGG16_Cifar10" \
    --epoch 200 \
    --lr 0.1 \
    --batch_size 128 \
    --weight_decay 5e-4 \
    --device cuda \
    --num_workers 4

# Train VGG16 on Cifar-100
echo ""
echo "[2/6] Training VGG16 on Cifar-100..."
python "$TRAIN_SCRIPT" \
    --model VGG16 \
    --dataset Cifar-100 \
    --path "$WEIGHT_DIR/VGG16_Cifar100" \
    --epoch 200 \
    --lr 0.1 \
    --batch_size 128 \
    --weight_decay 5e-4 \
    --device cuda \
    --num_workers 4

# Train VGG16 on ImageNet
echo ""
echo "[3/6] Training VGG16 on ImageNet..."
python "$TRAIN_SCRIPT" \
    --model VGG16 \
    --dataset ImageNet \
    --path "$WEIGHT_DIR/VGG16_ImageNet" \
    --epoch 90 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 1e-4 \
    --device cuda \
    --num_workers 8

# Train Deit on Cifar-10
echo ""
echo "[4/6] Training Deit on Cifar-10..."
python "$TRAIN_SCRIPT" \
    --model Deit \
    --dataset Cifar-10 \
    --path "$WEIGHT_DIR/Deit_Cifar10" \
    --epoch 200 \
    --lr 0.1 \
    --batch_size 128 \
    --weight_decay 5e-4 \
    --device cuda \
    --num_workers 4

# Train Deit on Cifar-100
echo ""
echo "[5/6] Training Deit on Cifar-100..."
python "$TRAIN_SCRIPT" \
    --model Deit \
    --dataset Cifar-100 \
    --path "$WEIGHT_DIR/Deit_Cifar100" \
    --epoch 200 \
    --lr 0.1 \
    --batch_size 128 \
    --weight_decay 5e-4 \
    --device cuda \
    --num_workers 4

# Train Deit on ImageNet
echo ""
echo "[6/6] Training Deit on ImageNet..."
python "$TRAIN_SCRIPT" \
    --model Deit \
    --dataset ImageNet \
    --path "$WEIGHT_DIR/Deit_ImageNet" \
    --epoch 300 \
    --lr 0.001 \
    --batch_size 256 \
    --weight_decay 0.05 \
    --device cuda \
    --num_workers 8

echo ""
echo "========================================"
echo "All training tasks completed!"
echo "Model weights saved to: $WEIGHT_DIR"
echo "========================================"
