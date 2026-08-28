#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dog vs. Non-Dog Verification Module

This module checks whether the ground-truth pet label and the predicted classifier label
represent a dog or a non-dog entity by querying the reference dog breeds dataset (dognames.txt).
"""


def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to indicate whether the pet image label
    and the classifier predicted label correspond to a dog.

    Appends two binary indicator elements:
      - Index 3: 1 if ground truth pet label is a dog, 0 otherwise
      - Index 4: 1 if classifier predicted label is a dog, 0 otherwise

    Parameters:
      results_dic (dict): Dictionary mapping image filenames to list of labels and metrics.
                          [pet_label, classifier_label, match_flag]
      dogfile (str)     : Path to the text file containing valid dog breed names.

    Returns:
      None: Modifies results_dic in-place.
    """
    # Load dog names into a hash set / dictionary for fast O(1) membership checks
    dognames_dic = dict()
    with open(dogfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            dogname = line.strip()
            if dogname and dogname not in dognames_dic:
                dognames_dic[dogname] = 1

    # Evaluate each entry in results dictionary
    for filename in results_dic:
        pet_label = results_dic[filename][0]
        classifier_label = results_dic[filename][1]

        # Check dog membership
        pet_is_dog = 1 if pet_label in dognames_dic else 0
        classifier_is_dog = 1 if classifier_label in dognames_dic else 0

        # Append [pet_is_dog, classifier_is_dog] at indices 3 and 4
        results_dic[filename].extend([pet_is_dog, classifier_is_dog])
