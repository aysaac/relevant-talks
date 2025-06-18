# Weights & Biases (wandb) Examples

This directory contains examples of using Weights & Biases (wandb) for experiment tracking in machine learning projects.

## MNIST Example

The `mnist_example.py` script demonstrates how to use wandb to track experiments when training a simple CNN model on the MNIST dataset.

### Features

- Loading and preprocessing the MNIST dataset
- Building a simple CNN model using PyTorch
- Tracking experiments with wandb
- Logging metrics, images, and model architecture
- Creating a confusion matrix visualization
- Saving and uploading the trained model

### Requirements

To run this example, you need to install the required packages:

```bash
pip install -r requirements.txt
```

You also need to have a wandb account. If you don't have one, you can sign up for free at [wandb.ai](https://wandb.ai).

### Usage

1. Make sure you're logged in to wandb:
   ```bash
   wandb login
   ```

2. Run the example script:
   ```bash
   python mnist_example.py
   ```

3. The script will:
   - Download the MNIST dataset (if not already downloaded)
   - Initialize a wandb run
   - Train a CNN model for 5 epochs
   - Log training and testing metrics to wandb
   - Save the trained model
   - Provide a link to view your experiment on the wandb dashboard

### Customization

You can modify the hyperparameters in the script to experiment with different configurations:

- `batch_size`: Number of samples per batch
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `momentum`: Momentum for SGD optimizer

### Visualizations

The example logs several visualizations to wandb:

- Sample images from the dataset
- Training and testing loss curves
- Testing accuracy
- Confusion matrix

These visualizations can be viewed on your wandb dashboard.