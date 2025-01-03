"""
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
import numpy as np
from pandas import DataFrame


class PCA:
    def __init__(self, n_components: int):
        """
        Ref: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        :param n_components: The number of dimensions to which the data must be reduced.
        """
        self.n_components: int = n_components

        self.top_eigenvectors = None  # these are the top n principal components
        self.explained_variance = None  # these are the top n eigenvalues
        self.total_sum_of_eigenvalues = None

    def _mean_center(self, dataset: DataFrame) -> DataFrame:
        return dataset - dataset.mean()

    def get_sorted_eigenvals_and_eigenvectors(self, dataset: DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the eigenvalues and eigenvectors sorted in descending order.
        :param dataset:
        :return:
        """
        eigenvalues, eigenvectors = np.linalg.eig(np.cov(dataset.T))

        # we may get complex numbers due to precision issues (that happened when I ran it the first time). So, discard the imaginary part.
        if np.iscomplexobj(eigenvalues):
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)

        # Now, sort the eigenvalues in descending order, and thereby sort their corresponding eigenvectors.
        # Reason being, the largest eigenvalue gives 1st principal component, 2nd largest gives 2nd principal component and so on.
        # Take the n largest eigenvalues and corresponding eigenvectors from them, and proceed. Store the top n principal components.
        # the following is vectorized.
        eigenvectors = eigenvectors[:,
                       np.argsort(-eigenvalues)]  # descending order of indices used to sort eigenvectors
        eigenvalues = -np.sort(-eigenvalues)  # descending order of sorting AFTER sorting eigenvectors

        return eigenvalues, eigenvectors

    def fit(self, dataset: DataFrame) -> 'PCA':
        """
        Obtains the principal components of our dataset.

        It does this by finding the covariance matrix.
        :return: The model itself.
        """
        # Mean center the data first (so that the covariance doesn't depend on mean, and so that it doesn't get distorted by the effect of means.)
        # We need the covariance matrix to show the true relationship between features.
        # Note that if the dataset is already standardized, then it has already been mean centered.
        dataset = self._mean_center(dataset)

        # Get the sorted eigenvalues and the eigenvectors.
        sorted_eigenvalues, sorted_eigenvectors = self.get_sorted_eigenvals_and_eigenvectors(dataset)
        self.top_eigenvectors = sorted_eigenvectors[:, :self.n_components]

        self.total_sum_of_eigenvalues = np.sum(sorted_eigenvalues)
        self.explained_variance = sorted_eigenvalues[:self.n_components]

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transforms the data using these principal components that were obtained from fit().
        :return: DataFrame of shape (n_samples, n_components)
        """
        X = self._mean_center(X)
        # Now, to project the data onto the new basis, we need the matrix product of transpose of eigenVectors with our original data.
        # We know that for 2 vectors x and y, x^T y = x.y and so, we will use this fact.
        return X.dot(self.top_eigenvectors)

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        return X.dot(self.top_eigenvectors.T)

    def checkPCA(self, originalData: DataFrame, projectedData: DataFrame, error_threshold: float = 1) -> bool:
        """
        :param originalData: The original, unreduced dataset.
        :param projectedData: The reduced dataset obtained via PCA.
        :param error_threshold: A threshold for numerical errors in calculation. 1 by default.
        :return: True, if this class reduces the dimensions appropriately. False otherwise.
        """
        if originalData.shape[0] != projectedData.shape[0] or projectedData.shape[1] != self.n_components:
            return False

        originalData = self._mean_center(originalData)
        reconstructed_data = self.inverse_transform(projectedData)
        originalData.columns = range(originalData.shape[1])

        reconstruction_error = np.mean(np.square(reconstructed_data - originalData))  # Mean Squared Error
        print(f"Reconstruction Error for {self.n_components}D PCA: {reconstruction_error}")

        return reconstruction_error <= error_threshold



