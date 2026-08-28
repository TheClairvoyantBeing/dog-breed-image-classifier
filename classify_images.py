#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification & Label Comparison Module

This module feeds images through the pretrained CNN classifier and compares
the predicted ImageNet label against the ground-truth pet label.
"""

import os
from classifier import classifier


def classify_images(images_dir, results_dic, model):
    """
    Classifies all images in results_dic using the specified CNN architecture,
    standardizes the model output, and evaluates whether the classification matches
    the ground truth.

    Appends two elements to each dictionary value list:
      - Index 1: Classifier predicted label (lowercase string, stripped)
      - Index 2: Match indicator (1 = match, 0 = no match)

    Parameters:
      images_dir (str)  : Directory path containing the images.
      results_dic (dict): Dictionary mapping image filenames to list of labels.
                          Initially contains [pet_image_label] at index 0.
      model (str)       : CNN architecture name ('resnet', 'alexnet', or 'vgg').

    Returns:
      None: Modifies results_dic in-place.
    """
    for filename in results_dic:
        # Build full image path
        image_path = os.path.join(images_dir, filename)

        # Run inference via classifier engine
        raw_prediction = classifier(image_path, model)

        # Standardize prediction: lowercase and trim whitespace
        model_label = raw_prediction.lower().strip()

        # Retrieve ground truth
        truth = results_dic[filename][0]

        # ImageNet classes often list comma-separated synonyms (e.g. 'beagle', 'basset hound')
        # We test whether the ground truth matches any individual synonym or substring
        synonyms = [term.strip() for term in model_label.split(',')]
        is_match = 1 if (truth in synonyms or truth in model_label) else 0

        # Append predicted label and match flag
        results_dic[filename].extend([model_label, is_match])
