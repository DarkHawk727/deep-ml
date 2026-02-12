# Deep ML

A repository for my solutions to the problems on [Deep-ML](https://www.deep-ml.com/), a site for LeetCode-style questions for machine learning and data science. For each problem, I decided to use either `numpy` or pure Python, depending on the type signature of the method, i.e. if the method takes in 2 `np.array`s, then I use `numpy`, else Python.

## Collections

> [!NOTE]
> Collections have duplicate questions.

1. **Deep Learning**
    1. Linear Algebra
        - [X] [Matrix-Vector Dot Product](src/001_matrix_vector_dot_product.py)
        - [X] [Transpose of a Matrix](src/002_transpose_of_a_matrix.py)
        - [X] [Dot Product Calculator](src/083_dot_product_calculator.py)
        - [X] [Scalar Multiplication of a Matrix](src/005_scalar_multiplication_of_a_matrix.py)
        - [X] [Calculate Cosine Similarity Between Vectors](src/076_calculate_cosine_similarity_between_vectors.py)
        - [X] [Calculate Mean by Row or Column](src/004_calculate_mean_by_row_or_column.py)
        - [X] [Calculate Eigenvalues of a Matrix](src/006_calculate_eigenvalues_of_a_matrix.py)
        - [X] [Calculate 2x2 Matrix Inverse](src/008_calculate_2x2_matrix_inverse.py)
        - [X] [Matrix time Matrix](src/009_matrix_times_matrix.py)
    2. Probability and Statistics
        - [X] [Poisson Distribution Probability Calculator](src/081_poisson_distribution_probability_calculator.py)
        - [X] [Binomial Distribution Probability](src/079_binomial_distribution_probability.py)
        - [X] [Normal Distribution PDF Calculator](src/080_normal_distribution_PDF_calculator.py)
        - [X] [Descriptive Statistics Calculator](src/078_descriptive_statistics_calculator.py)
        - [X] [Calculate Covariance Matrix](src/010_calculate_covariance_matrix.py)
    3. Optimization Techniques
        - [X] [Linear Regression Using Gradient Descent](src/015_linear_regression_using_gradient_descent.py)
        - [X] [Implement Gradient Descent Variants with MSE Loss](src/047_implement_gradient_descent_variants_with_MSE_loss.py)
        - [ ] Implement Adam Optimization Algorithm
        - [ ] Implement Lasso Regression using Gradient Descent
    4. Fundamentals of Neural Networks
        - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
        - [X] [Implementation of Log Softmax Function](src/039_implementation_of_log_softmax_function.py)
        - [X] [Sigmoid Activation Function](src/022_sigmoid_activation_function_understanding.py)
        - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
        - [X] [Leaky ReLU Activation Function](src/044_leaky_relu_activation_function.py)
        - [X] [Implement the PReLU Activation Function](src/098_implement_the_prelu_activation_function.py)
        - [X] [Single Neuron](src/024_single_neuron.py)
        - [ ] Implementing a Simple RNN
        - [ ] Implement a Long Short-Term Memory (LSTM) Network
        - [ ] Simple Convolutional 2D Layer
        - [ ] GPT-2 Text Generation
    5. Backpropagation
        - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
        - [X] [Implementing Basic Autograd Operations](src/026_implementing_basic_autograd_operations.py)
        - [ ] Implement a Simple RNN with Backpropagation Through Time (BPTT)
    6. LLM
        - [ ] Implement Self-Attention Mechanism
        - [ ] The Pattern Weaver's Code
        - [X] [Positional Encoding Calculator](src/085_positional_encoding_calculator.py)
        - [ ] Implement Multi-Head Attention
        - [ ] GPT-2 Text Generation
2. **DenseNet**
   - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
   - [ ] Simple Convolutional 2D Layer
   - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
   - [ ] Implement a Simple Residual Block with Shortcut Connection
   - [ ] Implement Global Average Pooling
   - [ ] Implement Batch Normalization for BCHW Input
   - [ ] Implement a Batch Dense Block with 2D Convolutions
3. **Linear Algebra**
   1. Vector Spaces
        - [X] [Matrix-Vector Dot Product](src/001_matrix_vector_dot_product.py)
        - [X] [Transpose of a Matrix](src/002_transpose_of_a_matrix.py)
        - [X] [Convert Vector to Diagonal Matrix](src/035_convert_vector_to_diagonal_matrix.py)
        - [X] [Dot Product Calculator](src/083_dot_product_calculator.py)
        - [ ] Find the Image of a Matrix Using Row Echelon Form
        - [X] [Calculate Cosine Similarity Between Vectors](src/076_calculate_cosine_similarity_between_vectors.py)
   2. Matrix Operations
        - [X] [Reshape Matrix](src/003_reshape_matrix.py)
        - [X] [Scalar Multiplication of a Matrix](src/005_scalar_multiplication_of_matrix.py)
        - [X] [Implement Compressed Row Sparse Matrix (CSR) Format Conversion](src/065_implement_compressed_row_sparse_matrix_format_conversion.py)
        - [ ] Implement Orthogonal Projection of a Vector onto a Line
        - [ ] Implement Compressed Column Sparse Matrix Format (CSC)
        - [X] [Transformation Matrix from Basis B to C](src/027_transformation_matrix_from_basis_B_to_C.py)
        - [X] [Matrix Transformation](src/007_matrix_transformation.py)
        - [X] [Calculate 2x2 Matrix Inverse](src/008_calculate_2x2_matrix_inverse.py)
        - [X] [Matrix times Matrix](src/009_matrix_times_matrix.py)
        - [ ] Implement Reduced Row Echelon Form (RREF) Function
   3. Eigenvalues and Eigenvectors
        - [X] [Calculate Eigenvalues of a Matrix](src/006_calculate_eigenvalues_of_a_matrix.py)
        - [X] [Solve Linear Equations using Jacobi Method](src/011_solve_linear_equations_using_jacobi_method.py)
        - [ ] Principal Component Analysis (PCA) Implementation
        - [ ] Singular Value Decomposition (SVD) of a 2x2 Matrix using Eigenvalues and Eigenvectors
   4. Matrix Factorization and Decomposition
        - [ ] 2D Translation of Matrix Implementation
        - [ ] Gauss-Seidel Method for Solving Linear Systems
        - [ ] Singular Value Decomposition (SVD)
        - [ ] Determinant of a 4x4 Matrix using Laplace's Expansion
4. **Machine Learning**
    1. Linear Algebra
        - [X] [Matrix-Vector Dot Product](src/001_matrix_vector_dot_product.py)
        - [X] [Transpose of a Matrix](src/002_transpose_of_a_matrix.py)
        - [X] [Dot Product Calculator](src/083_dot_product_calculator.py)
        - [X] [Scalar Multiplication of a Matrix](src/005_scalar_multiplication_of_matrix.py)
        - [X] [Calculate Cosine Similarity Between Vectors](src/076_calculate_cosine_similarity_between_vectors.py)
        - [X] [Calculate Mean by Row or Column](src/004_calculate_mean_by_row_or_column.py)
        - [X] [Calculate Eigenvalues of a Matrix](src/006_calculate_eigenvalues_of_a_matrix.py)
        - [X] [Calculate 2x2 Matrix Inverse](src/008_calculate_2x2_matrix_inverse.py)
        - [X] [Matrix time Matrix](src/009_matrix_times_matrix.py)
    2. Probability and Statistics
        - [X] [Poisson Distribution Probability Calculator](src/081_poisson_distribution_probability_calculator.py)
        - [X] [Binomial Distribution Probability](src/079_binomial_distribution_probability.py)
        - [X] [Normal Distribution PDF Calculator](src/080_normal_distribution_PDF_calculator.py)
        - [X] [Descriptive Statistics Calculator](src/078_descriptive_statistics_calculator.py)
        - [X] [Calculate Covariance Matrix](src/010_calculate_covariance_matrix.py)
    3. Optimization
        - [X] [Linear Regression Using Gradient Descent](src/015_linear_regression_using_gradient_descent.py)
        - [X] [Implement Gradient Descent Variants with MSE Loss](src/047_implement_gradient_descent_variants_with_MSE_loss.py)
        - [ ] Implement Adam Optimization Algorithm
        - [ ] Implement Lasso Regression using Gradient Descent
    4. Model Evaluation
        - [X] [Generate a Confusion Matrix for Binary Classification](src/075_generate_a_confusion_matrix_for_binary_classification.py)
        - [X] [Calculate Accuracy Score](src/036_calculate_accuracy_score.py)
        - [X] [Implement Precision Metric](src/046_implement_precision_metric.py)
        - [X] [Implement Recall Metric in Binary Classification](src/052_implement_recall_metric_in_binary_classification.py)
        - [X] I[mplement F-Score Calculation for Binary Classification](src/061_implement_f_score_calculation_for_binary_classification.py)
        - [X] [Calculate R-squared for Regression Analysis](src/069_calculate_r-squared_for_regression_analysis.py)
        - [ ] Calculate Mean Absolute Error
        - [X] [Calculate Root Mean Square Error (RMSE)](src/071_calculate_root_mean_square_error.py)
        - [X] [Implement K-Fold Cross-Validation](src/018_implement_k_fold_cross_validation.py)
        - [ ] Calculate Performance Metrics for a Classification Model
        - [X] [Implementation of Log Softmax Function](src/039_implementation_of_log_softmax_function.py)
        - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
    5. Classification & Regression Techniques
        - [X] [Linear Regression Using Normal Equation](src/014_linear_regression_using_normal_equation.py)
        - [X] [Linear Regression Using Gradient Descent](src/015_linear_regression_using_gradient_descent.py)
        - [ ] Binary Classification with Logistic Regression
        - [ ] [Calculate Jaccard Index for Binary Classification](src/072_calculate_jaccard_index_for_binary_classification.py)
        - [ ] Pegasos Kernel SVM Implementation
        - [ ] Implement AdaBoost Fit Method
        - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
    6. Unsupervised Learning
        - [X] [KL Divergence Between Two Normal Distributions](src/056_KL_divergence_between_two_normal_distributions.py)
        - [ ] Principal Component Analysis (PCA) Implementation
        - [X] [K-Means Clustering](src/017_k_means_clustering.py)
    7. Deep Learning
        - [X] [Single Neuron](src/024_single_neuron.py)
        - [X] [Sigmoid Activation Function Understanding](src/022_sigmoid_activation_function_understanding.py)
        - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
        - [X] [Implementation of Log Softmax](src/039_implementation_of_log_softmax_function.py)
        - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
        - [ ] Simple Convolutional 2d Layer
        - [ ] Implementation a Simple RNN
5. **ResNet**
    - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
    - [ ] Simple Convolutional 2D Layer
    - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
    - [ ] Implement a Simple Residual Block with Shortcut Connection
    - [ ] Implement Global Average Pooling
    - [ ] Implement Batch Normalization for BCHW Input
6. **Sparsely Gated MoE**
    - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
    - [X] [Single Neuron](src/024_single_neuron.py)
    - [ ] Calculate Computational Efficiency of MoE
    - [ ] Implement Noisy Top-K Gating Function
    - [ ] Implement a Sparse Mixture of Experts Layer
7. **Attention is All You Need**
    - [ ] Implement Self-Attention Mechanism
    - [ ] Implement Multi-Head Attention
    - [ ] Implement Masked Self-Attention
    - [ ] Implement Layer Normalization for Sequence Data
    - [X] [Positional Encoding Calculator](src/085_positional_encoding_calculator.py)
8. **Data Science I Interview Prep**
   1. Core Machine Learning Concepts
        - [X] [Linear Regression Using Gradient Descent](src/015_linear_regression_using_gradient_descent.py)
        - [ ] K-Means Clustering Implement Early Stopping Based on Validation Loss
        - [ ] Find the Best Gini-Based Split for a Binary Decision Tree
        - [ ] Implement K-Nearest Neighbours
   2. Data Processing
        - [X] [One-Hot Encoding of Nominal Values](src/034_one_hot_encoding_of_nominal_values.py)
        - [ ] Min-Max Normalization of Feature Values
        - [ ] Implement K-Fold Cross-Validation
        - [X] [Calculate Mean by Row or Column](src/004_calculate_mean_by_row_or_column.py)
        - [X] [Feature Scaling Implementation](src/016_feature_scaling_implementation.py)
   3. Deep Learning
        - [ ] Dropout Layer
        - [ ] Min-Max Normalization of Feature Values
        - [X] [Softmax Activation Function Function Implementation](src/023_softmax_activation_function_implementation.py)
        - [X] [Single Neuron](src/024_single_neuron.py)
        - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
   4. Model Evaluation & Metrics
        - [X] [Calculate F1 Score from Predict and True Labels](src/091_calculate_f1_score_from_predicted_and_true_labels.py)
        - [X] [Calculate Accuracy Score](src/036_calculate_accuracy_score.py)
        - [X] [Calculate Root Mean Square Error (RMSE)](src/071_calculate_root_mean_square_error.py)
        - [ ] Calculate Mean Absolute Error (MAE)
        - [X] [Implement Precision Metric](src/046_implement_precision_metric.py)
        - [ ] Detect Overfitting or Underfitting
        - [ ] ExponentialLR Learning Rate Scheduler
9. **Essense of Linear Algebra**
    1. Vectors
        - [X] [Scalar Multiplication of a Matrix](src/005_scalar_multiplication_of_matrix.py)
    2. Linear Combinations
        - [ ] Compute Orthonormal Basis for 2D Vectors
    3. Linear Transformations
        - [X] [Matrix-Vector Dot Product](src/001_matrix_vector_dot_product.py)
        - [X] [Dot Product Calculator](src/083_dot_product_calculator.py)
        - [X] [Transformation Matrix from Basis B to C](src/027_transformation_matrix_from_basis_B_to_C.py)
    4. Matrix Multiplication
        - [X] [Matrix times Matrix](src/009_matrix_times_matrix.py)
    5. Determinant
        - [ ] Determinant of a 4x4 Matrix using Laplace's Expansion
    6. Inverse Matrices
        - [X] [Calculate 2x2 Matrix Inverse](src/008_calculate_2x2_matrix_inverse.py)
        - [ ] Implement Reduced Row Echelon Form (RREF) Function
        - [ ] Find the Image of a Matrix Using Row Echelon Form  
    7. Cross Product
        - [ ] Compute the Cross Product of Two 3D Vectors
    8. Cramer's Rule
        - [ ] Solve System of Linear Equations Using Cramer's Rule  
    9. Change of Basis
        - [X] [Transformation Matrix from Basis B to C](src/027_transformation_matrix_from_basis_B_to_C.py)
    10. Eigenvector and Eigenvalues
        - [X] [Calculate Eigenvalues of a Matrix](src/006_calculate_eigenvalues_of_a_matrix.py)
10. **Micrograd Builder**
    - [X] [Sigmoid Activation Function Understanding](src/022_sigmoid_activation_function_understanding.py)
    - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
    - [X] [Implement Basic Autograd Operations](src/026_implementing_basic_autograd_operations.py)
    - [X] [Single Neuron](src/024_single_neuron.py)
    - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
11. **Optimizers**
    - [ ] [Implement Gradient Descent Variants with MSE Loss](src/047_implement_gradient_descent_variants_with_MSE_loss.py)
    - [ ] Implement Adam Optimization Algorithm
    - [ ] Adagrad Optimizer
    - [ ] Momentum Optimizer
    - [ ] Adamax Optimizer
    - [ ] Adadelta Optimizer
    - [ ] Nesterov Accelerated Gradient Optimizer
    - [ ] Find Captain Redbeard's Hidden Treasure
