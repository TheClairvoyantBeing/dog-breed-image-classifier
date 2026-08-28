#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classifier Benchmark Driver Script

Main orchestration script for evaluating and comparing pretrained CNN model architectures
(ResNet, AlexNet, VGG) on pet image classification and breed identification.

Usage:
    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
"""

from time import time

# Helper check functions for lab validation
from print_functions_for_lab_checks import (
    check_command_line_arguments,
    check_creating_pet_image_labels,
    check_classifying_images,
    check_classifying_labels_as_dogs,
    check_calculating_results
)

# Pipeline modules
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results


def main():
    # 1. Start execution timer
    start_time = time()

    # 2. Parse command-line arguments
    in_arg = get_input_args()
    check_command_line_arguments(in_arg)

    # 3. Create ground-truth pet image labels dictionary
    results = get_pet_labels(in_arg.dir)
    check_creating_pet_image_labels(results)

    # 4. Classify images using selected CNN model
    classify_images(in_arg.dir, results, in_arg.arch)
    check_classifying_images(results)

    # 5. Adjust results dictionary to classify labels as dog vs not-a-dog
    adjust_results4_isadog(results, in_arg.dogfile)
    check_classifying_labels_as_dogs(results)

    # 6. Calculate summary counts and percentage statistics
    results_stats = calculates_results_stats(results)
    check_calculating_results(results, results_stats)

    # 7. Print summary reports and misclassifications
    print_results(results, results_stats, in_arg.arch, print_incorrect_dogs=True, print_incorrect_breed=True)

    # 8. Compute and print total program runtime
    end_time = time()
    tot_time = end_time - start_time
    hours = int(tot_time / 3600)
    minutes = int((tot_time % 3600) / 60)
    seconds = int((tot_time % 3600) % 60)
    print(f"\n** Total Elapsed Runtime: {hours}:{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    main()
