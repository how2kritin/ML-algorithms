"""
Linear Regression using Gradient Descent
"""
from typing import Tuple

# References:
# https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec06-slides.pdf

import numpy as np
import pickle


class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, degree: int = 1, lmbda: float = 0, regularization_type: int = 2,
                 max_num_iterations: int = 1000, tolerance: float = 1e-6):
        """
        :param learning_rate: Learning Rate.
        :param degree: Degree of the curve being fit
        :param lmbda: Regularization hyperparameter, lambda
        :param regularization_type: L1 (1) or L2 (2) regularization. Only takes {1, 2} as values. L2 by default.
        :param max_num_iterations: Epoch; Maximum number of iterations before the gradient descent algorithm terminates. 1000 by default.
        :param tolerance: Default 1e-6. Tolerance for convergence; if the difference between MSE in 2 successive iterations is less than this tolerance, then break.
        """
        self.learning_rate: float = learning_rate  # learning rate
        self.degree: int = degree  # degree
        self.lmbda: float = lmbda  # this is lambda; a regularization hyperparameter.
        self.regularization_type: int = regularization_type  # L1 (1) or L2 (2) regularization. Only takes {1, 2} as values.
        self.max_num_iterations: int = max_num_iterations  # epoch; maximum number of iterations
        self.tolerance: int = tolerance  # convergence tolerance

        self.weights: np.ndarray = None  # Will be set later during training
        self.bias: int = None  # Will be set later during training

        # for tracking metrics during gradient descent
        self.MSE_history: list = []
        self.residual_stddev_history: list = []
        self.residual_var_history: list = []
        self.predicted_vals_history: list = []

    def _generate_polynomial_feature_map(self, X: np.ndarray):
        """
        To generate a polynomial feature map of degree 'degree'.
        :return:
        """
        X_poly_features = X.reshape(-1, 1)  # infer first dim, and set 2nd dime to 1 (make it a 2D array)

        # essentially, appending columns of higher degree to a copy of X.
        for deg in range(2, self.degree + 1):
            X_poly_features = np.append(X_poly_features, (X ** deg).reshape(-1, 1), axis=1)
        return X_poly_features

    def get_metrics(self) -> tuple[list, list, list, list]:
        """
        Returns MSE, standard deviation and variance of residuals from training.
        Also return the predicted values from each iteration of training.
        :return:
        """
        return self.MSE_history, self.residual_stddev_history, self.residual_var_history, self.predicted_vals_history

    def save_model(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump({'learning_rate': self.learning_rate, 'degree': self.degree, 'lmbda': self.lmbda,
                         'regularization_type': self.regularization_type, 'max_num_iterations': self.max_num_iterations,
                         'tolerance': self.tolerance, 'weights': self.weights, 'bias': self.bias}, file)

    def load_model(self, filename: str) -> None:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.learning_rate = data['learning_rate']
            self.degree = data['degree']
            self.lmbda = data['lmbda']
            self.regularization_type = data['regularization_type']
            self.max_num_iterations = data['max_num_iterations']
            self.tolerance = data['tolerance']
            self.weights = data['weights']
            self.bias = data['bias']

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        To fit the linear regression model using gradient descent, on a curve of degree 'degree' and with a
        learning rate 'learning_rate'.
        :param X_train: A numpy array.
        :param y_train: A numpy array.
        :return:
        """
        X_poly_features = self._generate_polynomial_feature_map(X_train)
        n, m = X_poly_features.shape

        # initially, both, weights and bias are set to 0 in gradient descent.
        self.weights = np.zeros(m)  # vector of weights (linear coefficients)
        self.bias = 0

        prev_MSE = float('inf')

        for _ in range(self.max_num_iterations):
            y_predicted = np.dot(X_poly_features, self.weights) + self.bias
            diff_residual = (y_predicted - y_train)
            MSE = np.mean(diff_residual ** 2)
            self.MSE_history.append(MSE)
            self.residual_stddev_history.append(np.std(diff_residual))
            self.residual_var_history.append(np.var(diff_residual))
            self.predicted_vals_history.append(y_predicted)

            if abs(prev_MSE - MSE) < self.tolerance:
                break

            prev_MSE = MSE

            regulariser = (2 * self.learning_rate * self.lmbda * self.weights) if self.regularization_type == 2 else \
                (self.learning_rate * self.lmbda * np.sign(self.weights))

            self.bias -= (2 * self.learning_rate / n) * np.sum(diff_residual)
            self.weights -= ((2 * self.learning_rate / n) * np.dot(X_poly_features.T, diff_residual) + regulariser)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise Exception("The model hasn't been fit yet!")

        X_poly_features = self._generate_polynomial_feature_map(X)
        return np.dot(X_poly_features, self.weights) + self.bias
