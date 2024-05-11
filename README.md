# Advanced Statistical Learning Techniques in Image Classification

This repository contains the projects I completed for the "Fundamentals of Statistical Learning" course. Each part of the project demonstrates the application of advanced statistical learning techniques to classify images using Support Vector Machines (SVM) and Convolutional Neural Networks (CNN) on the MNIST and CIFAR-10 datasets.

## Project Overview

### Part 1: Feature Extraction, Density Estimation, and Bayesian Classification
- **Dataset**: Modified MNIST dataset, focusing on digits "3" and "7".
- **Tasks**:
  - **Feature Extraction**: Calculated skewness and pixel ratio features from 28x28 pixel images.
  - **Density Estimation**: Employed Maximum Likelihood Estimation (MLE) to estimate distribution parameters for Bayesian classification.
  - **Classification**: Implemented Bayesian Decision Theory, achieving error rates of 22.2% and 22.1% on training and testing sets, respectively.

### Part 2: Experimenting with SVM
- **Dataset**: Custom dataset with 50 categories and 6,619 samples total (4,786 training, 1,833 testing).
- **Implementation**: Used libSVM to perform multi-class classification.
- **Results**:
  - Achieved classification accuracies of 10.78% to 28.78% using individual features.
  - Improved accuracy to 44.72% by combining features through classifier fusion.

### Part 3: Deep Learning with CNN
- **Dataset**: CIFAR-10, consisting of 60,000 32x32 color images across 10 classes.
- **Configuration**: Developed a CNN with 12 hidden layers, including ReLU activations, dropout, and batch normalization.
- **Performance**:
  - Initial model achieved a baseline test accuracy of 83.06%.
  - Various experiments with learning rates and batch sizes provided insights into optimal settings, with the best test accuracy recorded at 85.4%.

## Key Technologies
- **Python**: Primary programming language used for implementing and testing the models.
- **TensorFlow & Keras**: Utilized for building and training convolutional neural networks.
- **libSVM**: Employed for support vector machine classification tasks.
- **SciPy**: Used for data manipulation and feature extraction.

## Repository Structure
- `FSL_Baseline.ipynb`: Baseline model for the CNN experiments.
- `FSL_Project_Part_2.ipynb`: Notebook containing SVM experiments and results.
- `reports/`: Contains detailed PDF reports for each project part, summarizing methodologies, results, and observations.


