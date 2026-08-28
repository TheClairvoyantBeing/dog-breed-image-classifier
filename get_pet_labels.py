#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pet Image Ground Truth Label Extractor

This module extracts the true identity of each pet by parsing image filenames.
Filenames follow standard conventions (e.g., 'Boston_terrier_02259.jpg') where
words separated by underscores represent the label, followed by image numbers.
"""

from os import listdir


def get_pet_labels(image_dir):
    """
    Scans the given image directory and creates a dictionary mapping each image
    filename to its ground-truth label extracted from the filename.

    Formatting rules:
      - Convert to lowercase
      - Split on underscores
      - Retain only alphabetic words (dropping numbers and extensions)
      - Strip leading and trailing whitespace
      - Ignore system/hidden files starting with '.' (e.g., .DS_Store)

    Parameters:
      image_dir (str): Relative or absolute path to the directory of images.

    Returns:
      dict: Dictionary with format:
            { 'filename.jpg': ['ground truth pet label'] }
    """
    in_files = listdir(image_dir)
    results_dic = dict()

    for filename in in_files:
        # Skip hidden files and macOS metadata files
        if filename.startswith('.'):
            continue

        # Extract words from filename, lowercased
        lower_name = filename.lower()
        word_list = lower_name.split('_')

        # Accumulate only alphabetical parts of the pet name
        pet_label = ' '.join([word for word in word_list if word.isalpha()]).strip()

        # Insert into results dictionary with warning for duplicates
        if filename not in results_dic:
            results_dic[filename] = [pet_label]
        else:
            print(f"** Warning: Duplicate image file found: {filename}")

    return results_dic
