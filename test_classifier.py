#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier Test Script

Quick verification script to test the classifier engine on a sample image.

Usage:
    python test_classifier.py
"""

from classifier import classifier


def main():
    test_image = "pet_images/Collie_03797.jpg"
    model = "vgg"

    print(f"Running classifier on '{test_image}' using architecture '{model}'...")
    prediction = classifier(test_image, model)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
