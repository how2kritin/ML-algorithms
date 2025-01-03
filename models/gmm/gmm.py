"""
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
"""
import numpy as np
from typing import Self
from pandas import DataFrame
from scipy.stats import multivariate_normal
from models.k_means.k_means import K_Means

class GMM:
    def __init__(self, n_components: int = 1, max_iter: int = 100, tol: float = 1e-6, random_state: int = None):
        """
        :param n_components: Number of mixture components. 1 by default.
        :param max_iter: Maximum number of EM iterations to perform. 100 by default.
        :param tol: Convergence tolerance. EM iterations will stop when the lower bound average gain is below this threshold. 1e-6 by default.
        :param random_state: Random state to make the random initialization deterministic. Not set by default.
        """
        self.n_components: int = n_components
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.random_state: int = random_state

        # will be set later during fitting of the model
        self.mean_vector: np.ndarray = np.array([])
        self.covariance_matrices: np.ndarray = np.array([])
        self.pi: np.ndarray = np.array([])

        # AIC and BIC computations
        self.aic: float = None
        self.bic: float = None

    def _get_log_responsibilities(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            np.log(self.pi[k]) + multivariate_normal(mean=self.mean_vector[k], cov=self.covariance_matrices[k]).logpdf(
                X)
            for k in range(self.n_components)
        ]).T

    def _compute_information_criteria(self, X: np.ndarray, log_likelihood: float) -> None:
        """
        Calculate AIC and BIC for the model. Run this only after the model has been fit!
        :param X:
        :param log_likelihood:
        """
        n_samples, n_features = X.shape
        n_parameters = self.n_components * (n_features + n_features * (n_features + 1) / 2) + (self.n_components - 1)

        self.aic = 2 * n_parameters - 2 * log_likelihood
        self.bic = np.log(n_samples) * n_parameters - 2 * log_likelihood

    def _init_parameters(self, X: DataFrame) -> None:
        # Using k-means++'s initialization to initialize the parameters.
        kmm = K_Means(n_clusters=self.n_components, random_state=self.random_state).init_kmeans_plus_plus(X)
        self.mean_vector = kmm.getCentroids()
        labels = kmm.predict(X)  # labels after initialising the centroids
        self.covariance_matrices = np.zeros((self.n_components, X.shape[1], X.shape[1]))
        for k in range(self.n_components):
            cluster_points = X[labels == k]
            if cluster_points.shape[0] > 1:
                self.covariance_matrices[k] = np.cov(cluster_points, rowvar=False)
                self.covariance_matrices[k] += np.eye(X.shape[1]) * 1e-6  # to make the covariance matrices symmetric positive definite.
            else:
                self.covariance_matrices[k] = np.identity(X.shape[1]) * 1e-6  # for the cases with only 1 point in the cluster.
        counts = np.bincount(labels, minlength=self.n_components)
        self.pi = counts / np.sum(counts)

    def fit(self, X: DataFrame) -> Self:
        """
        Initializing parameters using K-Means++.
        :param X:
        :return:
        """
        self._init_parameters(X)
        X = X.values
        n_samples, n_features = X.shape

        # iterate until convergence:
        previous_log_likelihood = float(-np.inf)
        for _ in range(self.max_iter):
            # first, E-Step: Evaluating the responsibilities given current parameters:
            responsibilities = self.getMembership(X)

            # next, M-Step: Re-estimate the parameters given current responsibilities:
            N_k = np.maximum(responsibilities.sum(axis=0), 1e-6)  # just to make sure that it isn't 0.
            self.mean_vector = (responsibilities.T @ X) / N_k[:, np.newaxis]
            for k in range(self.n_components):
                X_centered = X - self.mean_vector[k]
                self.covariance_matrices[k] = (X_centered.T @ (X_centered * responsibilities[:, k][:, np.newaxis])) / N_k[k]
                self.covariance_matrices[k] += np.eye(n_features) * 1e-6  # adding a small value to the diagonal to ensure that it is positive definite (and not semi-definite)
            self.pi = N_k / n_samples

            curr_log_likelihood = self.getLikelihood(X)
            if abs(curr_log_likelihood - previous_log_likelihood) <= self.tol:
                break
            previous_log_likelihood = curr_log_likelihood

        # now that the model has been fit, compute the information criteria
        self._compute_information_criteria(X, previous_log_likelihood)
        return self

    def getParams(self) -> dict:
        return {"Mean Vector": self.mean_vector, "Covariance Matrices": self.covariance_matrices, "Mixing Coefficients": self.pi}

    def getMembership(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the responsibilities matrix.
        :param X:
        :return:
        """
        # getting log probs first, and then the actual exp probs.
        log_probs = self._get_log_responsibilities(X)
        log_probs_max = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - log_probs_max)
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

    def getLikelihood(self, X: np.ndarray) -> float:
        """
        Returns the log-likelihood.
        :param X:
        :return:
        """
        # using the log-sum-exponent method here, since it is more numerically stable (doesn't just rise to inf due to our multiplications)
        log_likelihoods = self._get_log_responsibilities(X)
        log_prob_max = np.max(log_likelihoods, axis=1, keepdims=True)
        exp_log_prob = np.exp(log_likelihoods - log_prob_max)
        sum_exp_log_prob = np.sum(exp_log_prob, axis=1)
        return np.sum(np.log(sum_exp_log_prob) + log_prob_max.ravel())  # .ravel() is to flatten the array.

    def predict(self, X: DataFrame) -> np.ndarray:
        """
        Returns the cluster assignments (most likely).
        :param X:
        :return:
        """
        X = X.values
        cluster_assignments = np.argmax(self.getMembership(X), axis=1)
        return cluster_assignments
