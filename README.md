# Fashion MNIST Classification Using Convolutional Neural Networks (CNNs) and Transfer Learning with VGG16

## Overview

This repository contains the implementation and comparison of two deep learning approaches for classifying fashion articles in the Fashion MNIST dataset. The first approach uses a custom Convolutional Neural Network (CNN), while the second leverages the VGG16 model, a pre-trained network, for transfer learning.

## Part 1: Custom CNN Model

### Implementation Details

- **Data Preprocessing**: The Fashion MNIST dataset, consisting of 60,000 training images and 10,000 test images, is normalized to have pixel values in the range [0, 1]. The dataset includes 10 classes of fashion items.
- **Model Architecture**: The CNN model comprises several convolutional layers, each followed by batch normalization, max-pooling, and dropout layers to enhance performance and reduce overfitting. The model ends with fully connected layers for classification.
- **Training**: The model is trained with Adam optimizer, using sparse categorical crossentropy as the loss function. Early stopping is implemented to halt training when the validation loss ceases to decrease, preventing overfitting.

### Results

The custom CNN model achieved a test accuracy of **92.07%**, demonstrating its effectiveness in classifying fashion items. The model correctly predicted 14 out of 15 items in a sample test set visualization.

## Part 2: Transfer Learning with VGG16

### Implementation Details

- **Data Preprocessing**: Similar to Part 1, but images are resized to 32x32x3 to match the input shape expected by VGG16. Labels are converted to one-hot encoded format.
- **Model Architecture**: VGG16 is used as the base model with its top layers removed. Custom dense layers are added on top of VGG16 to tailor the model to the Fashion MNIST dataset.
- **Training**: The model is compiled and trained using the same strategy as in Part 1, with adjustments to accommodate the different input size and the one-hot encoded labels.

### Results

The transfer learning approach using VGG16 achieved a test accuracy of **92.04%**. This performance is slightly lower than the custom CNN but confirms the viability of transfer learning for this task.

## Conclusions

Both models demonstrated high accuracy in classifying fashion items from the Fashion MNIST dataset. The custom CNN model slightly outperformed the VGG16 transfer learning model, showcasing the potential of tailored architectures for specific datasets. However, the VGG16 model also showed strong performance, highlighting the effectiveness of transfer learning, especially when computational resources or labeled data are limited.

This project illustrates the strengths of both custom CNNs and transfer learning approaches in tackling image classification problems, providing a foundation for further exploration and optimization.

## Usage

To replicate the results or experiment with the models, ensure you have TensorFlow and Keras installed. Run the provided Jupyter Notebook, which contains all necessary code for data preprocessing, model building, training, and evaluation. Adjust the hyperparameters as needed to explore their impact on model performance.
