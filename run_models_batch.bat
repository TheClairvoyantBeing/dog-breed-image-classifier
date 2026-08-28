@echo off
REM =========================================================================
REM Batch Script: Benchmark CNN Architectures on Pet Images (Windows)
REM =========================================================================

if not exist "results" mkdir results

echo Running ResNet on pet_images...
py -3.11 check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt > results\resnet_pet-images.txt

echo Running AlexNet on pet_images...
py -3.11 check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > results\alexnet_pet-images.txt

echo Running VGG on pet_images...
py -3.11 check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt > results\vgg_pet-images.txt

echo Batch evaluation complete! Results saved in results\
