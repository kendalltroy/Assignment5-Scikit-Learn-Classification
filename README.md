# Assignment5-Scikit-Learn-Classification

# Purpose of Program
The purpose of this program is to compare 3 different classification models by training and testing them on Scikit Learn's breast cancer data set. The best model will be chosen based on its accuracy, confusion matrix, and classification report.

# Input
The expected input of this program is the Scikit Learn's breast cancer dataset. The class attributes are 0 as Malignant and 1 as Benign.

# Type of Execution & Methods
The program performs the following types of execution:
1. Deterministic Execution: because all of the random seeds are set to 42, the output of the program is predictable and consistent. This also ensures the models are comparable.
2. Model Execution: This program executes three models: K-Nearest Neighbors, Linear Perceptron, and Support Vector Machine. Each of the models go through a sequential flow of loading data, splitting data, scaling, training, and testing.
3. Console-Based Output: All outputs are printed in a standard method (accuracy, confusion matrix, and classification report.

# Limitations
1. Small dataset: the dataset is only 569 samples, which is fine for demonstration purposes. However, it is questionable how much it reflects real world scenarios
2. Potential overfitting: Due to the reletively small dataset, it is possible the models are overfit.
3. No clinical context: without proper medical knowledge, machine learning models should not be used to diagnose a patient with cancer.
