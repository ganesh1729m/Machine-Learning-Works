# Programming Assignment - 04: Neural Networks

This programming assignment focuses on building a neural network from scratch to perform classification tasks, specifically on the MNIST dataset. It includes various parts where different activation functions, optimization algorithms, and regularization techniques are implemented and compared. The final part involves comparing the performance of deep neural network models with linear classification models on both linearly separable and non-linearly separable data.

## Instructions

1. **Plagiarism is strictly prohibited.**
2. **Delayed submissions will be penalized with a scaling factor of 0.5 per day.**
3. **Please DO NOT use any machine learning libraries unless otherwise specified.**

## Part - (1): Develop a Neural Network-Based Classification Network from Scratch

### 1. Load MNIST Data and Create Train-Test Splits
- **Dataset:** MNIST digit dataset consisting of 70,000 images.
- **Train-Test Split:** 60,000 images for training and 10,000 images for testing.
- **Code:** Provided for downloading the data and creating train-test splits.

### 2. Design a Simple Classification Network
- **Architecture:** Three-layer feed-forward neural network with:
  - **Hidden Layers:** 512 nodes each.
  - **Output Layer:** 10 nodes.
- **Activation Functions:**
  - **Hidden Layers:** ReLU.
  - **Output Layer:** Softmax.
- **Training:**
  - **Flatten Images:** 28x28 images to 784-dimensional vectors.
  - **Parameter Initialization:** Random initialization.
  - **Optimization Algorithm:** Stochastic Gradient Descent (SGD).

### 3. Evaluate the Performance of the Classification Network
- **Metrics:** Loss and accuracy.
- **Evaluation:** Feed-forward MNIST data through the trained network and compute the loss and accuracy.

## Part - (2): Understanding Activation Functions

### Experiment with Different Activation Functions
- **Functions:** Sigmoid, Tanh, ReLU, LeakyReLU.
- **Training:** Using SGD.
- **Report:** Accuracy on MNIST test data and observations.

## Part - (3): Understanding Optimization Algorithms

### Train with Adam Optimization Algorithm
- **Best Activation Function:** From Part - (2).
- **Comparison:** Accuracy of networks trained with SGD and Adam.
- **Report:** Observations.

## Part - (4): Understanding Regularization Methods

### Techniques to Reduce Overfitting
1. **Weight Regularization:** Add regularization term to the classification loss.
2. **Dropout:** Probability of 0.2. Experiment with different probabilities.
3. **Early Stopping:** Stop training when overfitting starts.

### Report
- **Accuracies:** For each regularization technique.
- **Observations:** Detailed comparison.

## Part - (5): Comparison with Linear Classifiers

### 1. Linearly Separable Data
- **Classes:** Gaussian distribution with specified mean vectors and covariance matrices.
- **Samples:** 4500 per class for training, 500 for testing.

### 2. Non-Linearly Separable Data
- **Dataset:** Given code to generate binary classification data.
- **Split:** 90% for training, 10% for testing.

### 3. Linear Classification Models
- **Logistic Regression:** Iterative reweighted least squares approach.
- **Function:** `Logistic_Regression` to find optimal weights `w` and make predictions.
- **Evaluation:** Accuracy on test data.
- **Visualization:** Decision regions.

### 4. Deep Neural Network-Based Classification Models
- **Architecture:** Three-layer feed-forward neural network.
- **Activation Functions:**
  - **Hidden Layers:** ReLU.
  - **Output Layer:** Sigmoid.
- **Training:** Given training data, plot second layer activation potentials, and evaluate performance on test data.

### 5. Comparison
- **Models:** Linear classification vs. deep neural network-based classification.
- **Report:** Observations and comparisons.

## Conclusion

This assignment covers the implementation and comparison of various neural network techniques, activation functions, optimization algorithms, and regularization methods. By following the above structure, you can understand the steps taken to solve the classification tasks and compare the effectiveness of different methods.


