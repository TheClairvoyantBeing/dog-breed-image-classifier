#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretrained CNN Inference Engine

This module encapsulates torchvision deep learning models (ResNet-18, AlexNet, VGG-16)
pre-trained on ImageNet-1k, and handles image preprocessing and class prediction.
"""

import ast
import os
import warnings
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as tv_models

# Suppress minor torchvision deprecation warnings
warnings.filterwarnings('ignore')


class LazyModels(dict):
    """
    Dictionary proxy that lazily instantiates and evaluates torchvision models
    only when requested, avoiding unnecessary memory and download overhead.
    """
    def __init__(self):
        super().__init__()
        self._models = {}

    def __getitem__(self, key):
        if key not in self._models:
            if key == 'resnet':
                try:
                    m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
                except Exception:
                    m = tv_models.resnet18(pretrained=True)
            elif key == 'alexnet':
                try:
                    m = tv_models.alexnet(weights=tv_models.AlexNet_Weights.DEFAULT)
                except Exception:
                    m = tv_models.alexnet(pretrained=True)
            elif key == 'vgg':
                try:
                    m = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
                except Exception:
                    m = tv_models.vgg16(pretrained=True)
            else:
                raise KeyError(f"Unsupported CNN architecture: {key}. Choose 'resnet', 'alexnet', or 'vgg'.")
            m.eval()
            self._models[key] = m
        return self._models[key]


# Model container instance
models = LazyModels()

# Load ImageNet class ID to human-readable label mapping
LABEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet1000_clsid_to_human.txt')
with open(LABEL_FILE, 'r', encoding='utf-8') as classes_file:
    imagenet_classes_dict = ast.literal_eval(classes_file.read())


def classifier(img_path, model_name):
    """
    Classifies a single image using the specified CNN model architecture.

    Parameters:
      img_path (str)   : Path to the target image file.
      model_name (str) : Architecture name ('resnet', 'alexnet', or 'vgg').

    Returns:
      str: Human-readable ImageNet class prediction label.
    """
    # Open image and ensure 3-channel RGB format
    with Image.open(img_path) as img_pil:
        img_rgb = img_pil.convert('RGB')

        # Standard ImageNet normalization and cropping pipeline
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = preprocess(img_rgb)
        img_tensor.unsqueeze_(0)

    # Perform inference without tracking gradients
    model = models[model_name]
    with torch.no_grad():
        output = model(img_tensor)

    # Extract class index with highest probability
    pred_idx = output.data.numpy().argmax()

    return imagenet_classes_dict[pred_idx]
