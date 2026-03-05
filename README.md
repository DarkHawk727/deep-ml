# Deep ML

A repository for my solutions to the problems on [Deep-ML](https://www.deep-ml.com/), a site for LeetCode-style questions for machine learning and data science. For each problem, I decided to use either `numpy` or pure Python, depending on the type signature of the method, i.e. if the method takes in 2 `np.array`s, then I use `numpy`, else Python.

## Collections

> [!NOTE]
> Collections have duplicate questions.

1. **AlexNet**
    - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
    - [X] [Simple Convolutional 2D Layer](src/041_simple_convolutional_2d_layer.py)
    - [ ] Overlapping Max Pooling
    - [ ] PCA Colour Augmentation
    - [ ] Dropout Layer
2. **Attention is All You Need**
    - [ ] Implement Self-Attention Mechanism
    - [ ] Implement Multi-Head Attention
    - [ ] Implement Masked Self-Attention
    - [ ] Implement Layer Normalization for Sequence Data
    - [X] [Positional Encoding Calculator](src/085_positional_encoding_calculator.py)
3. **Deep Learning**
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
        - [X] [Implement Adam Optimization Algorithm](src/049_implement_adam_optimization_algorithm.py)
        - [X] [Implement Lasso Regression using ISTA](src/050_implement_lasso_regression_using_ISTA.py)
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
        - [X] [Simple Convolutional 2D Layer](src/041_simple_convolutional_2d_layer.py)
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
4. **DeepSeek R1**
    - [ ] Implement the GRPO Objective Function
    - [ ] Group Relative Advantage for GRPO
    - [ ] KL Divergence Estimator for GRPO
    - [ ] Pass@k and Majority Voting Evaluation Metrics
    - [ ] Knowledge Distillation Loss
5. **DenseNet**
    - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
    - [X] [Simple Convolutional 2D Layer](src/041_simple_convolutional_2d_layer.py)
    - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
    - [ ] Implement a Simple Residual Block with Shortcut Connection
    - [ ] Implement Global Average Pooling
    - [ ] Implement Batch Normalization for BCHW Input
    - [ ] Implement a Batch Dense Block with 2D Convolutions
6. **GPT 243**
   1. Autograd Engine (Value Class & Backpropagation)
        - [X] [Implementing Basic Autograd Operations](src/026_implementing_basic_autograd_operations.py)
        - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
   2. Lab: Autograd
        - [ ] MNIST: Classification Loss (with Gradient)
   3. Tokenization & Embeddings
        - [ ] Character-Level Tokenizer (stoi/itos/BOS)
        - [ ] Learned Positional Embeddings
   4. Lab: Tokenization
        - [ ] Build a Tokenizer for Language Modelling
   5. Core Building Blocks (Linear, Softmax, RMSNorm)
        - [X] [Matrix-Vector Dot Product](src/001_matrix_vector_dot_product.py)
        - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
        - [ ] Implement RMSNorm (Root Mean Square Layer Normalization)
   6. Lab: Build a Neural Network from Scratch
        - [ ] MNIST: Build Neural NEtwork from Scratch (NumPy Only)
   7. Multi-Head Attention & KV Cache
        - [ ] Implement Self-Attention Mechanism
        - [ ] Implement Masked Self-Attention
        - [ ] Implement Multi-Head Attention
        - [ ] KV Cache for Efficient Autogregressive Attention
   8. Lab: Attention
        - [ ] Design Your Own Attention Mechanism
   9. Transformer Block (Residuals, MLP, Activations)
        - [ ] Implement a Simple Residual Block with Shortcut Connection
        - [ ] Implement Position-wise Feed-Forwards Block with Residual and Dropout
        - [ ] Implement the Square ReLU Activation Function
   10. Lab: Activation Function
        - [X] [Design Your Own Activation Function](src)
   11. Loss Functions & Cross-Entropy
        - [ ] Compute Multi-class Cross-Entropy Loss
        - [X] [Implementation of Log Softmax Function](src/039_implementation_of_log_softmax_function.py)
   12. Adam Optimizer & Learning Rate Schedule
        - [X] [Implement Adam Optimization Algorithm](src/049_implement_adam_optimization_algorithm.py)
        - [ ] Linear Learning Rate Decay
   13. Lab: Optimizer
        - [ ] Design Your Own Optimizer (NumPy)
   14. Training Loop (Putting It All Together)
        - [ ] Calculate Number of Parameters in Neural Network
   15. Lab: Full Training Loop
        - [ ] Build a Digit Classifier from Scratch
   16. Inference & Text Generation
        - [ ] Temperature Sampling
7. **LLM Evaluation Methods**
   1. Multiple Choice Benchmarks
        - [ ] MMLU Letter-Matching Evaluation
        - [ ] MMLU Log-Probability Scoring
   2. Verifier-Based Evaluation
        - [ ] Boxed Answer Extraction for Math Benchmarks
        - [ ] Math Answer Verification with Equivalence Checking
        - [ ] Code Execution Verifier for Programming Benchmarks
   3. Preference Leaderboards
        - [ ] Elo Rating System for Model Comparison
        - [ ] Bradley-Terry Model for Pairwise Rankings
   4. LLM-as-a-Judge
        - [ ] Rubric-Based LLM Judge Evaluation
        - [ ] Pairwise Preference Judge for LLM Comparison
   5. Other Measures Mentioned in the Post
        - [ ] BLEU Score for Text Generation
        - [ ] Calculate PReplexity for Language Models
        - [ ] Compute Multi-class Cross-Entropy Loss
8. **Linear Algebra**
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
        - [X] [Principal Component Analysis (PCA) Implementation](src/019_principal_component_analysis_implementation.py)
        - [ ] Singular Value Decomposition (SVD) of a 2x2 Matrix using Eigenvalues and Eigenvectors
   4. Matrix Factorization and Decomposition
        - [ ] 2D Translation of Matrix Implementation
        - [ ] Gauss-Seidel Method for Solving Linear Systems
        - [ ] Singular Value Decomposition (SVD)
        - [ ] Determinant of a 4x4 Matrix using Laplace's Expansion
9. **Machine Learning**
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
        - [X] [Implement Adam Optimization Algorithm](src/049_implement_adam_optimization_algorithm.py)
        - [X] [Implement Lasso Regression using ISTA](src/050_implement_lasso_regression_using_ISTA.py)
    4. Model Evaluation
        - [X] [Generate a Confusion Matrix for Binary Classification](src/075_generate_a_confusion_matrix_for_binary_classification.py)
        - [X] [Calculate Accuracy Score](src/036_calculate_accuracy_score.py)
        - [X] [Implement Precision Metric](src/046_implement_precision_metric.py)
        - [X] [Implement Recall Metric in Binary Classification](src/052_implement_recall_metric_in_binary_classification.py)
        - [X] [Implement F-Score Calculation for Binary Classification](src/061_implement_f_score_calculation_for_binary_classification.py)
        - [X] [Calculate R-squared for Regression Analysis](src/069_calculate_r-squared_for_regression_analysis.py)
        - [X] [Calculate Mean Absolute Error (MAE)](src/093_calculate_mean_absolute_error.py)
        - [X] [Calculate Root Mean Square Error (RMSE)](src/071_calculate_root_mean_square_error.py)
        - [X] [Implement K-Fold Cross-Validation](src/018_implement_k_fold_cross_validation.py)
        - [X] [Calculate Performance Metrics for a Classification Model](src/077_calculate_performance_metrics_for_a_classification_model.py)
        - [X] [Implementation of Log Softmax Function](src/039_implementation_of_log_softmax_function.py)
        - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
    5. Classification & Regression Techniques
        - [X] [Linear Regression Using Normal Equation](src/014_linear_regression_using_normal_equation.py)
        - [X] [Linear Regression Using Gradient Descent](src/015_linear_regression_using_gradient_descent.py)
        - [X] [Binary Classification with Logistic Regression](src/104_binary_classification_with_logistic_regression.py)
        - [X] [Calculate Jaccard Index for Binary Classification](src/072_calculate_jaccard_index_for_binary_classification.py)
        - [ ] Pegasos Kernel SVM Implementation
        - [ ] Implement AdaBoost Fit Method
        - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
    6. Unsupervised Learning
        - [X] [KL Divergence Between Two Normal Distributions](src/056_KL_divergence_between_two_normal_distributions.py)
        - [X] [Principal Component Analysis (PCA) Implementation](src/019_principal_component_analysis_implementation.py)
        - [X] [K-Means Clustering](src/017_k_means_clustering.py)
    7. Deep Learning
        - [X] [Single Neuron](src/024_single_neuron.py)
        - [X] [Sigmoid Activation Function Understanding](src/022_sigmoid_activation_function_understanding.py)
        - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
        - [X] [Implementation of Log Softmax](src/039_implementation_of_log_softmax_function.py)
        - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
        - [X] [Simple Convolutional 2d Layer](src/041_simple_convolutional_2d_layer.py)
        - [ ] Implementation a Simple RNN
10. **Metadata Normalization (MDN)**
    1. Mathematical Prerequisites
        - [X] [Linear Regression Using Normal Equation](src/014_linear_regression_using_normal_equation.py)
        - [ ] Implement Orthogonal Projection of a Vector onto a Line
    2. Normalization Baselines (What MDN Improves Upon)
        - [ ] Implement Batch Normalization for BCHW Input
        - [ ] Implement Group Normalization
    3. Core MDN Concepts
        - [ ] Implement Code MDN Residualization
        - [ ] Distance Correlation for Measuring Metadata Dependence
    4. Advanced MDN (Handling Confounding)
        - [ ] MDN with Label Collinearity Control
    5. Lab
        - [ ] Feature Deconfounder for Biased Image Data
11. **ResNet**
    - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
    - [X] [Simple Convolutional 2D Layer](src/041_simple_convolutional_2d_layer.py)
    - [X] [Implement ReLU Activation Function](src/042_implement_relu_activation_function.py)
    - [ ] Implement a Simple Residual Block with Shortcut Connection
    - [ ] Implement Global Average Pooling
    - [ ] Implement Batch Normalization for BCHW Input
12. **Sparsely Gated MoE**
    - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
    - [X] [Single Neuron](src/024_single_neuron.py)
    - [ ] Calculate Computational Efficiency of MoE
    - [ ] Implement Noisy Top-K Gating Function
    - [ ] Implement a Sparse Mixture of Experts Layer
13. **Data Science I Interview Prep**
    1. Core Machine Learning Concepts
         - [X] [Linear Regression Using Gradient Descent](src/015_linear_regression_using_gradient_descent.py)
         - [ ] K-Means Clustering Implement Early Stopping Based on Validation Loss
         - [ ] Find the Best Gini-Based Split for a Binary Decision Tree
         - [ ] Implement K-Nearest Neighbours
    2. Data Processing
         - [X] [One-Hot Encoding of Nominal Values](src/034_one_hot_encoding_of_nominal_values.py)
         - [ ] Min-Max Normalization of Feature Values
         - [X] [Implement K-Fold Cross-Validation](src/018_implement_k_fold_cross_validation.py)
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
         - [X] [Calculate Mean Absolute Error (MAE)](src/093_calculate_mean_absolute_error.py)
         - [X] [Implement Precision Metric](src/046_implement_precision_metric.py)
         - [ ] Detect Overfitting or Underfitting
         - [ ] ExponentialLR Learning Rate Scheduler
14. **Essense of Linear Algebra**
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
        - [X] [Compute the Cross Product of Two 3D Vectors](src/118_compute_the_cross_product_of_two_3D_vectors.py)
    8. Cramer's Rule
        - [ ] Solve System of Linear Equations Using Cramer's Rule  
    9. Change of Basis
        - [X] [Transformation Matrix from Basis B to C](src/027_transformation_matrix_from_basis_B_to_C.py)
    10. Eigenvector and Eigenvalues
        - [X] [Calculate Eigenvalues of a Matrix](src/006_calculate_eigenvalues_of_a_matrix.py)
15. **Micrograd Builder**
    - [X] [Sigmoid Activation Function Understanding](src/022_sigmoid_activation_function_understanding.py)
    - [X] [Softmax Activation Function Implementation](src/023_softmax_activation_function_implementation.py)
    - [X] [Implement Basic Autograd Operations](src/026_implementing_basic_autograd_operations.py)
    - [X] [Single Neuron](src/024_single_neuron.py)
    - [X] [Single Neuron with Backpropagation](src/025_single_neuron_with_backpropogation.py)
16. **Optimizers**
    - [X] [Implement Gradient Descent Variants with MSE Loss](src/047_implement_gradient_descent_variants_with_MSE_loss.py)
    - [X] [Implement Adam Optimization Algorithm](src/049_implement_adam_optimization_algorithm.py)
    - [ ] Adagrad Optimizer
    - [X] [Momentum Optimizer](src/146_momentum_optimizer.py)
    - [ ] Adamax Optimizer
    - [ ] Adadelta Optimizer
    - [ ] Nesterov Accelerated Gradient Optimizer
    - [X] [Find Captain Redbeard's Hidden Treasure](src/127_find_captain_redbeards_hidden_treasure.py)
