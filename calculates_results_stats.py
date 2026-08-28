#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Results Calculator Module

This module analyzes the completed classification results dictionary and computes
summary statistics, including total counts and accuracy percentages across dog detection,
breed classification, and non-dog filtering.
"""


def calculates_results_stats(results_dic):
    """
    Computes summary metrics and percentage accuracies from the results dictionary.

    Statistics computed:
      - n_images            : Total count of processed images
      - n_dogs_img          : Count of true dog images
      - n_notdogs_img       : Count of true non-dog images
      - n_match             : Count of exact label matches
      - n_correct_dogs      : Count of true dogs correctly predicted as dogs
      - n_correct_notdogs   : Count of non-dogs correctly predicted as non-dogs
      - n_correct_breed     : Count of true dogs where breed was correctly identified
      - pct_match           : Percentage of correct label matches over all images
      - pct_correct_dogs    : Percentage of correct dog classifications
      - pct_correct_breed   : Percentage of correct breed classifications among dogs
      - pct_correct_notdogs : Percentage of correct non-dog classifications

    Parameters:
      results_dic (dict): Dictionary with format:
                          { filename: [pet_label, classifier_label, is_match,
                                       pet_is_dog, classifier_is_dog] }

    Returns:
      dict: Dictionary of all calculated count and percentage statistics.
    """
    stats_dic = dict()

    # Total image count
    stats_dic['n_images'] = len(results_dic)

    # Initialize count accumulators
    stats_dic['n_dogs_img'] = 0
    stats_dic['n_notdogs_img'] = 0
    stats_dic['n_match'] = 0
    stats_dic['n_correct_dogs'] = 0
    stats_dic['n_correct_notdogs'] = 0
    stats_dic['n_correct_breed'] = 0

    # Tally counts across results dictionary
    for filename, entry in results_dic.items():
        is_match = entry[2]
        pet_is_dog = entry[3]
        classifier_is_dog = entry[4]

        # Overall label match
        if is_match == 1:
            stats_dic['n_match'] += 1

        # Process Dog vs Non-Dog ground truth
        if pet_is_dog == 1:
            stats_dic['n_dogs_img'] += 1

            # Correctly identified as a dog
            if classifier_is_dog == 1:
                stats_dic['n_correct_dogs'] += 1

            # Correctly identified dog breed (dog + match)
            if is_match == 1:
                stats_dic['n_correct_breed'] += 1
        else:
            stats_dic['n_notdogs_img'] += 1

            # Correctly identified as NOT a dog
            if classifier_is_dog == 0:
                stats_dic['n_correct_notdogs'] += 1

    # Calculate percentage accuracies
    n_images = stats_dic['n_images']
    n_dogs = stats_dic['n_dogs_img']
    n_notdogs = stats_dic['n_notdogs_img']

    stats_dic['pct_match'] = (stats_dic['n_match'] / n_images * 100.0) if n_images > 0 else 0.0
    stats_dic['pct_correct_dogs'] = (stats_dic['n_correct_dogs'] / n_dogs * 100.0) if n_dogs > 0 else 0.0
    stats_dic['pct_correct_breed'] = (stats_dic['n_correct_breed'] / n_dogs * 100.0) if n_dogs > 0 else 0.0
    stats_dic['pct_correct_notdogs'] = (stats_dic['n_correct_notdogs'] / n_notdogs * 100.0) if n_notdogs > 0 else 0.0

    return stats_dic
