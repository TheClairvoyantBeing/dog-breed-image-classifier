#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Summary Formatter & Printer Module

This module formats and prints the benchmark summary report, including overall image
counts, model percentage accuracies, and detailed breakdowns of misclassified dogs
or dog breeds when requested.
"""


def print_results(results_dic, results_stats_dic, model, 
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints a formatted summary table of classification results and optionally displays
    detailed misclassification reports.

    Parameters:
      results_dic (dict)        : Dictionary mapping filenames to labels and classification flags.
      results_stats_dic (dict)  : Dictionary containing computed counts and percentage metrics.
      model (str)               : Name of the CNN architecture used ('resnet', 'alexnet', 'vgg').
      print_incorrect_dogs (bool): When True, prints instances of misclassified dogs/not-dogs.
      print_incorrect_breed (bool): When True, prints instances of misclassified dog breeds.

    Returns:
      None: Outputs directly to console.
    """
    # Header
    print(f"\n\n*** Results Summary for CNN Model Architecture {model.upper()} ***")
    print(f"{'N Images':20}: {results_stats_dic['n_images']:3d}")
    print(f"{'N Dog Images':20}: {results_stats_dic['n_dogs_img']:3d}")
    print(f"{'N Not-Dog Images':20}: {results_stats_dic['n_notdogs_img']:3d}")

    # Percentages
    print(" ")
    for key, value in results_stats_dic.items():
        if key.startswith('pct'):
            print(f"{key:20}: {value:.1f}%")

    # Optional report: Misclassified Dogs vs Not-Dogs
    has_dog_misclassifications = (
        (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs']) 
        != results_stats_dic['n_images']
    )
    if print_incorrect_dogs and has_dog_misclassifications:
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for filename, data in results_dic.items():
            # A dog misclassification occurs when ground truth and predicted dog flags disagree
            if sum(data[3:]) == 1:
                print(f"Real: {data[0]:>26}   Classifier: {data[1]:>30}")

    # Optional report: Misclassified Dog Breeds
    has_breed_misclassifications = (
        results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed']
    )
    if print_incorrect_breed and has_breed_misclassifications:
        print("\nINCORRECT Dog Breed Assignment:")
        for filename, data in results_dic.items():
            # Both pet and classifier agree it is a dog, but breed label does not match
            if sum(data[3:]) == 2 and data[2] == 0:
                print(f"Real: {data[0]:>26}   Classifier: {data[1]:>30}")
