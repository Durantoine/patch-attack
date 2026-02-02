# Architecture Configurations

This directory contains YAML configuration files for different model architectures used in the patch attack experiments.

## Available Architectures

- `resnet50.yaml`: ResNet-50 architecture
- `vgg16.yaml`: VGG-16 architecture
- `efficientnet.yaml`: EfficientNet architecture

## Usage

To use a specific architecture, specify the config file in your training command:

```bash
python src/train.py --config configs/architectures/resnet50.yaml
```
