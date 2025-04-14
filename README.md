# Vision Transformer (ViT) Implementation

## Overview

This project implements a Vision Transformer (ViT) model for image classification on either the MNIST or CIFAR-10 datasets. The implementation follows the architecture described in the original paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al. This paper extends the core principles introduced by the paper which focuses on the performance of vision transformers when pre-trained on large datasets like JFT-300M which contains millions of images. However, this paper investigates the application of Vision Transformers (ViT) to image classification tasks on smaller datasets without relying on large-scale pre-training. This implementation doesn't require extensive computational resources. 
The notebook provides a complete pipeline from data loading and preprocessing to model training, evaluation, and visualization. It includes detailed model analysis tools to understand the architecture's complexity and performance.

## Features

- Flexible configuration for either MNIST or CIFAR-10 datasets
- Complete ViT implementation with:
  - Patch embedding layer
  - Position embeddings
  - Transformer encoder blocks
  - Multi-head self-attention
  - MLP classification head
- Detailed model analysis including:
  - Parameter counting
  - Layer-wise summary
  - Computational complexity
- Training with:
  - Adam optimizer
  - Cosine annealing learning rate schedule
  - Gradient clipping
- Visualization tools for:
  - Training history
  - Sample predictions
  - Performance metrics

## Requirements

The notebook requires standard Python machine learning libraries:
- PyTorch 1.8+
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- tqdm
### Note: All these packages are pre-installed on Google Colab

## Hardware Recommendations

For best performance, it is recommended to use a GPU-accelerated environment:
- **Google Colab**: Use a T4 or higher GPU (set via `Runtime > Change runtime type`)
- **Local machine**: Any CUDA-compatible GPU with at least 4GB memory

## Usage

1. **Dataset Selection**: 
   - Set `DATASET_CHOICE = "cifar10"` or `"mnist"` at the top of the notebook
   - Or run with command line arguments: `python script.py --dataset cifar10`

2. **Training**:
   - The model will automatically train for the specified number of epochs
   - Training progress and metrics are displayed during execution

3. **Evaluation**:
   - The notebook automatically evaluates on the test set after training
   - Provides accuracy metrics and visualizations of sample predictions

4. **Visualization**:
   - Training/validation loss curves
   - Accuracy progression
   - Sample predictions with true vs predicted labels

## Configuration

All hyperparameters are managed through the `Config` class, including:
- Model architecture parameters (embedding dimensions, number of heads, etc.)
- Training parameters (learning rate, batch size, epochs)
- Dataset-specific settings

## Results

After training, the notebook displays:
- Final test accuracy
- Training time
- Visualizations of model predictions
- Training history plots

## Extensions

The implementation can be easily extended to:
- Other image classification datasets
- Different model sizes by adjusting configuration parameters
- Additional visualization or analysis tools

## Note

For optimal performance on Google Colab, ensure you're using a GPU runtime and have sufficient storage space for the datasets (especially CIFAR-10). All required libraries are standard or pre-installed in Google Colab.
