# Numerical Methods

Welcome to the **Numerical Methods** repository! This repository contains implementations and experiments on optimization techniques, machine learning models, and mathematical problem-solving methods. Each file in the repository focuses on a specific topic, detailed below:

## File Descriptions

### 1. `MNIST_Optimization_and_Deep_Learning_Model_Training.py`
- **Topic**: MNIST Optimization and Deep Learning Model Training
- **Description**: This script demonstrates the implementation of a deep learning model using TensorFlow and JAX. It optimizes hyperparameters and evaluates performance on the MNIST dataset using grid search with the W&B (Weights & Biases) platform.

### 2. `Optimization_on_the_Rosenbrock_Function.py`
- **Topic**: Optimization on the Rosenbrock Function
- **Description**: Compares optimization methods such as Lion Optimizer, ADAM, and Gradient Descent with Momentum. The Rosenbrock function serves as the benchmark for these optimization trajectories.

### 3. `Constrained_Optimization_Using_Lagrange_Multipliers.py`
- **Topic**: Constrained Optimization Using Lagrange Multipliers
- **Description**: Implements a minimization method with Lagrange multipliers for constrained optimization problems. Includes customizable objective and constraint functions.

### 4. `Test_Scenarios_and_Minimization_Techniques.py`
- **Topic**: Test Scenarios and Minimization Techniques
- **Description**: Contains test cases for constrained optimization problems. The code implements penalization methods to enforce constraints during minimization.

### 5. `Polynomial_Regression_Application_and_Modeling.ipynb`
- **Topic**: Polynomial Regression Application and Modeling
- **Description**: Analyzes data and builds a polynomial regression model using Python. This notebook is a great resource for understanding regression techniques (inferred from the filename).

### 6. `Matrix_Inversion_and_Optimization_Approaches_with_JAX.py`
- **Topic**: Matrix Inversion and Optimization with JAX
- **Description**: Demonstrates matrix inversion using both analytical and optimization approaches in JAX. Includes performance comparisons and practical insights.

### 7. `Data_Compression_and_Latent_Space_Analysis_Using_Autoencoders_and_tSNE.py`
- **Topic**: Autoencoders and Latent Space Analysis
- **Description**: Implements an autoencoder for dimensionality reduction on the MNIST dataset. Visualizes the latent space using t-SNE and evaluates the reconstruction quality.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- Required libraries: `jax`, `numpy`, `tensorflow`, `tensorflow-datasets`, `flax`, `optax`, `matplotlib`, `sklearn`

Install dependencies via:
```bash
pip install -r requirements.txt
```

### Running the Scripts
- For `.py` files, run the scripts directly in the terminal using:
  ```bash
  python <filename>.py
  ```
- For the Jupyter Notebook (`.ipynb`), open it using Jupyter Notebook or JupyterLab and execute the cells interactively.

### Example
To visualize optimization trajectories in `Optimization_on_the_Rosenbrock_Function.py`, run:
```bash
python Optimization_on_the_Rosenbrock_Function.py
```
This will generate plots comparing different optimization methods.

## Repository Structure
```
Numerical-Methods/
├── MNIST_Optimization_and_Deep_Learning_Model_Training.py
├── Optimization_on_the_Rosenbrock_Function.py
├── Constrained_Optimization_Using_Lagrange_Multipliers.py
├── Test_Scenarios_and_Minimization_Techniques.py
├── Polynomial_Regression_Application_and_Modeling.ipynb
├── Matrix_Inversion_and_Optimization_Approaches_with_JAX.py
├── Data_Compression_and_Latent_Space_Analysis_Using_Autoencoders_and_tSNE.py
└── README.md
```

## Contributions
Contributions are welcome! If you'd like to improve the existing code or add new numerical methods, feel free to fork this repository and submit a pull request.

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
