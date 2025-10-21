# Deep ML

A repository for my solutions to the problems on [Deep-ML](https://www.deep-ml.com/), a site for LeetCode-style questions for machine learning and data science. For each problem, I decided to use either `numpy` or pure Python, depending on the type signature of the method, i.e. if the method takes in 2 `np.array`s, then I use `numpy`, else Python.

## Collections

> [!NOTE]
> Collections have duplicate questions.

1. **Deep Learning**
    1. Linear Algebra
        - [X] Matrix-Vector Dot Product
        - [X] Transpose of a Matrix
        - [X] Dot Product Calculator
        - [X] Scalar Multiplication of a Matrix
        - [X] Calculate Cosine Similarity Between Vectors
        - [X] Calculate Mean by Row or Column
        - [X] Calculate Eigenvalues of a Matrix
        - [X] Calculate 2x2 Matrix Inverse
        - [X] Matrix time Matrix
    2. Probability and Statistics
        - [X] Poisson Distribution Probability Calculator
        - [X] Binomial Distribution Probability
        - [X] Normal Distribution PDF Calculator
        - [X] Descriptive Statistics Calculator
        - [X] Calculate Covariance Matrix
    3. Optimization Techniques
        - [X] Linear Regression Using Gradient Descent
        - [ ] Implement Gradient Descent Variants with MSE Loss
        - [ ] Implement Adam Optimization Algorithm
        - [ ] Implement Lasso Regression using Gradient Descent
    4. Fundamentals of Neural Networks
        - [X] Softmax Activation Function Implementation
        - [X] Implementation of Log Softmax Function
        - [X] Sigmoid Activation Function
        - [X] Implement ReLU Activation Function
        - [X] Leaky ReLU Activation Function
        - [X] Implement the PReLU Activation Function
        - [X] Single Neuron
        - [ ] Implementing a Simple RNN
        - [ ] Implement a Long Short-Term Memory (LSTM) Network
        - [ ] Simple Convolutional 2D Layer
        - [ ] GPT-2 Text Generation
    5. Backpropagation
        - [ ] Single Neuron with Backpropagation
        - [ ] Implementing Basic Autograd Operations
        - [ ] Implement a Simple RNN with Backpropagation Through Time (BPTT)
    6. LLM
        - [ ] Implement Self-Attention Mechanism
        - [ ] The Pattern Weaver's Code
        - [ ] Positional Encoding Calculator
        - [ ] Implement Multi-Head Attention
        - [ ] GPT-2 Text Generation
2. **DenseNet**
   - [ ] Single Neuron with Backpropagation
   - [ ] Simple Convolutional 2D Layer
   - [X] Implement ReLU Activation Function
   - [ ] Implement a Simple Residual Block with Shortcut Connection
   - [ ] Implement Global Average Pooling
   - [ ] Implement Batch Normalization for BCHW Input
   - [ ] Implement a Batch Dense Block with 2D Convolutions
3. **Linear Algebra**
   1. Vector Spaces
        - [X] Matrix-Vector Dot Product
        - [X] Transpose of a Matrix
        - [ ] Convert Vector to Diagonal Matrix
        - [X] Dot Product Calculator
        - [ ] Find the Image of a Matrix Using Row Echelon Form
        - [X] Calculate Cosine Similarity Between Vectors
   2. Matrix Operations
        - [X] Reshape Matrix
        - [X] Scalar Multiplication of a Matrix
        - [ ] Implement Compressed Row Sparse Matrix (CSR) Format Conversion
        - [ ] Implement Orthogonal Projection of a Vector onto a Line
        - [ ] Implement Compressed Column Sparse Matrix Format (CSC)
        - [X] Transformation Matrix from Basis B to C
        - [X] Matrix Transformation
        - [X] Calculator 2x2 Matrix Inverse
        - [X] Matrix times Matrix
        - [ ] Implement Reduced Row Echelon Form (RREF) Function
   3. Eigenvalues and Eigenvectors
        - [X] Calculate Eigenvalues of a Matrix
        - [X] Solve Linear Equations using Jacobi Method
        - [ ] Principal Component Analysis (PCA) Implementation
        - [ ] Singular Value Decomposition (SVD) of a 2x2 Matrix using Eigenvalues and Eigenvectors
   4. Matrix Factorization and Decomposition
        - [ ] 2D Translation of Matrix Implementation
        - [ ] Gauss-Seidel Method for Solving Linear Systems
        - [ ] Singular Value Decomposition (SVD)
        - [ ] Determinant of a 4x4 Matrix using Laplace's Expansion
4. **Machine Learning**
    1. Linear Algebra
        - [X] Matrix-Vector Dot Product
        - [X] Transpose of a Matrix
        - [X] Dot Product Calculator
        - [X] Scalar Multiplication of a Matrix
        - [X] Calculate Cosine Similarity Between Vectors
        - [X] Calculate Mean by Row or Column
        - [X] Calculate Eigenvalues of a Matrix
        - [X] Calculate 2x2 Matrix Inverse
        - [X] Matrix time Matrix
    2. Probability and Statistics
        - [X] Poisson Distribution Probability Calculator
        - [X] Binomial Distribution Probability
        - [X] Normal Distribution PDF Calculator
        - [X] Descriptive Statistics Calculator
        - [X] Calculate Covariance Matrix
    3. Optimization
        - [X] Linear Regression Using Gradient Descent
        - [ ] Implement Gradient Descent Variants with MSE Loss
        - [ ] Implement Adam Optimization Algorithm
        - [ ] Implement Lasso Regression using Gradient Descent
    4. Model Evaluation
        - [ ] Generate a Confusion Matrix for Binary Classification
        - [X] Calculate Accuracy Score
        - [X] Implement Precision Metric
        - [ ] Implement Recall Metric in Binary Classification
        - [ ] Implement F-Score Calculation for Binary Classification
        - [ ] Calculate R-squared for Regression Analysis
        - [ ] Calculate Mean Absolute Error
        - [ ] Calculate Root Mean Square Error
        - [ ] Implement K-Fold Cross-Validation
        - [ ] Calculate Performance Metrics for a Classification Model
        - [X] Implementation of Log Softmax Function
        - [X] Implement ReLU Activation Function
    5. Classification & Regression Techniques
        - [X] Linear Regression Using Normal Equation
        - [X] Linear Regression Using Gradient Descent
        - [ ] Binary Classification with Logistic Regression
        - [ ] Calculate Jaccard Index for Binary Classification
        - [ ] Pegasos Kernel SVM Implementation
        - [ ] Implement AdaBoost Fit Method
        - [X] Softmax Activation Function Implementation
    6. Unsupervised Learning
        - [ ] KL Divergence Between Two Normal Distributions
        - [ ] Principal Component Analysis (PCA) Implementation
        - [ ] K-Means Clustering
    7. Deep Learning
        - [X] Single Neuron
        - [X] Sigmoid Activation Function Understanding
        - [X] Softmax Activation Function Implementation
        - [X] Implementation of Log Softmax
        - [X] Implement ReLU Activation Function
        - [ ] Simple Convolutional 2d Layer
        - [ ] Implementation a Simple RNN
5. **ResNet**
    - [ ] Single Neuron with Backpropagation
    - [ ] Simple Convolutional 2D Layer
    - [X] Implement ReLU Activation Function
    - [ ] Implement a Simple Residual Block with Shortcut Connection
    - [ ] Implement Global Average Pooling
    - [ ] Implement Batch Normalization for BCHW Input
6. **Sparsely Gated MoE**
    - [X] Softmax Activation Function Implementation
    - [X] Single Neuron
    - [ ] Calculate Computational Efficiency of MoE
    - [ ] Implement Noisy Top-K Gating Function
    - [ ] Implement a Sparse Mixture of Experts Layer
7. **Attention is All You Need**
    - [ ] Implement Self-Attention Mechanism
    - [ ] Implement Multi-Head Attention
    - [ ] Implement Masked Self-Attention
    - [ ] Implement Layer Normalization for Sequence Data
    - [ ] Positional Encoding Calculator
8. **Data Science I Interview Prep**
   1. Core Machine Learning Concepts
        - [X] Linear Regression Using Gradient Descent
        - [ ] K-Means Clustering Implement Early Stopping Based on Validation Loss
        - [ ] Find the Best Gini-Based Split for a Binary Decision Tree
        - [ ] Implement K-Nearest Neighbours
   2. Data Processing
        - [X] One-Hot Encoding of Nominal Values
        - [ ] Min-Max Normalization of Feature Values
        - [ ] Implement K-Fold Cross-Validation
        - [X] Calculate Mean by Row or Column
        - [X] Feature Scaling Implementation
   3. Deep Learning
        - [ ] Dropout Layer
        - [ ] Min-Max Normalization of Feature Values
        - [X] Softmax Activation Function Function Implementation
        - [X] Single Neuron
        - [X] Implement ReLU Activation Function
   4. Model Evaluation & Metrics
        - [ ] Calculate F1 Score from Predict and True Labels
        - [ ] Calculate Accuracy Score
        - [ ] Calculate Root Mean Square Error (RMSE)
        - [ ] Calculate Mean Absolute Error (MAE)
        - [X] Implement Precision Metric
        - [ ] Detect Overfitting or Underfitting
        - [ ] ExponentialLR Learning Rate Scheduler
9. **Essense of Linear Algebra**
    1. Vectors
        - [X] Scalar Multiplication of a Matrix
    2. Linear Combinations
        - [ ] Compute Orthonormal Basis for 2D Vectors
    3. Linear Transformations
        - [X] Matrix-Vector Dot Product
        - [X] Dot Product Calculator
        - [X] Transformation Matrix from Basis B to C
    4. Matrix Multiplication
        - [X] Matrix times Matrix
    5. Determinant
        - [X] Determinant of a 4x4 Matrix using Laplace's Expansion
    6. Inverse Matrices
        - [X] Calculate 2x2 Matrix Inverse
        - [ ] Implement Reduced Row Echelon Form (RREF) Function
        - [ ] Find the Image of a Matrix Using Row Echelon Form  
    7. Cross Product
        - [ ] Compute the Cross Product of Two 3D Vectors
    8. Cramer's Rule
        - [ ] Solve System of Linear Equations Using Cramer's Rule  
    9. Change of Basis
        - [X] Transformation Matrix from Basis B to C
    10. Eigenvector and Eigenvalues
        - [X] Calculate Eigenvalues of a Matrix  
10. **Micrograd Builder**
    - [X] Sigmoid Activation Function Understanding
    - [X] Softmax Activation Function Implementation
    - [ ] Implement Basic Autograd Operations
    - [X] Single Neuron
    - [ ] Single Neuron with Backpropagation
11. **Optimizers**
    - [ ] Implement Gradient Descent Variants with MSE Loss
    - [ ] Implement Adam Optimization Algorithm
    - [ ] Adagrad Optimizer
    - [ ] Momentum Optimizer
    - [ ] Adamax Optimizer
    - [ ] Adadelta Optimizer
    - [ ] Nesterov Accelerated Gradient Optimizer
    - [ ] Find Captain Redbeard's Hidden Treasure
