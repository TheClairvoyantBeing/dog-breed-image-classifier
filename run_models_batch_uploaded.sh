#!/usr/bin/env bash
# =========================================================================
# Batch Script: Benchmark CNN Architectures on Uploaded Images (Linux / macOS)
# =========================================================================

mkdir -p results

echo "Running ResNet on uploaded_images..."
python check_images.py --dir uploaded_images/ --arch resnet --dogfile dognames.txt > results/resnet_uploaded-images.txt

echo "Running AlexNet on uploaded_images..."
python check_images.py --dir uploaded_images/ --arch alexnet --dogfile dognames.txt > results/alexnet_uploaded-images.txt

echo "Running VGG on uploaded_images..."
python check_images.py --dir uploaded_images/ --arch vgg --dogfile dognames.txt > results/vgg_uploaded-images.txt

echo "Batch evaluation complete! Results saved in results/"
