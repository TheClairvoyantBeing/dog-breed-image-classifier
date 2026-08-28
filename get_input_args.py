#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Argument Parser Module

This module defines the CLI interface for the image classifier application,
allowing users to specify the input image directory, CNN model architecture,
and dog names reference dataset with sensible defaults.
"""

import argparse


def get_input_args():
    """
    Retrieves and parses command-line arguments passed to the script.

    Arguments parsed:
      --dir     : Path to the folder containing image files (default: 'pet_images/')
      --arch    : Pretrained CNN architecture to use ('vgg', 'resnet', 'alexnet') (default: 'vgg')
      --dogfile : Path to the text file containing valid dog breed names (default: 'dognames.txt')

    Returns:
      argparse.Namespace: Object containing parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Classify pet images using a pre-trained CNN and benchmark accuracy vs runtime."
    )

    # Image directory path
    parser.add_argument(
        "--dir",
        type=str,
        default="pet_images/",
        help="Path to folder containing images to classify (default: 'pet_images/')"
    )

    # Model architecture selection
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg",
        choices=["resnet", "alexnet", "vgg"],
        help="CNN model architecture: resnet, alexnet, or vgg (default: 'vgg')"
    )

    # Dog names reference file
    parser.add_argument(
        "--dogfile",
        type=str,
        default="dognames.txt",
        help="Text file containing valid dog names (default: 'dognames.txt')"
    )

    return parser.parse_args()
