#!/bin/bash
#Presented by KeJi
#Date ： 2026-02-12

# T-MLP Training Script for Linux
# This script trains VGG16 and Deit models on Cifar-10 and Cifar-100 datasets

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
echo "[1/4] Training VGG16 on Cifar-10..."
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
echo "[2/4] Training VGG16 on Cifar-100..."
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

# Train Deit on Cifar-10
echo ""
echo "[3/4] Training Deit on Cifar-10..."
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
echo "[4/4] Training Deit on Cifar-100..."
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

echo ""
echo "========================================"
echo "All training tasks completed!"
echo "Model weights saved to: $WEIGHT_DIR"
echo "========================================"
