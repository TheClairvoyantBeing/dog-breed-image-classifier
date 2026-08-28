#!/usr/bin/env bash
# =========================================================================
# Batch Script: Benchmark CNN Architectures on Pet Images (Linux / macOS)
# =========================================================================

mkdir -p results

echo "Running ResNet on pet_images..."
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt > results/resnet_pet-images.txt

echo "Running AlexNet on pet_images..."
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > results/alexnet_pet-images.txt

echo "Running VGG on pet_images..."
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt > results/vgg_pet-images.txt

echo "Batch evaluation complete! Results saved in results/"
