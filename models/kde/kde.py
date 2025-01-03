"""
Ref: https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
"""
import numpy as np
from typing import Literal, Callable
import matplotlib.pyplot as plt

class KDEKernels:
    def __init__(self, kernel: Literal['gaussian', 'box', 'triangular']):
        self.function: Callable
        match kernel:
            case 'gaussian':
                self.function = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
            case 'box':
                # f(x) = 1/2 if |x| <= 1, else 0.
                self.function = lambda x: np.where(np.abs(x) <= 1, 0.5, 0)
            case 'triangular':
                # f(x) = (1 - |x|) if |x| <= 1, else 0.
                self.function = lambda x: np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)


class KDE:
    def __init__(self, bandwidth: float = 1.0, kernel: Literal['gaussian', 'box', 'triangular'] = 'gaussian'):
        """
        :param bandwidth: The bandwidth (h) of the kernel. Determines the estimate's smoothness.
        :param kernel: gaussian by default. Must be one of {'gaussian', 'box', 'triangular'}.
        """
        self.bandwidth = bandwidth
        self.kernel_type = kernel
        self.kernel = KDEKernels(self.kernel_type).function

        self.data: np.ndarray = None

    def fit(self, X: np.ndarray) -> 'KDE':
        self.data = X
        return self

    def predict(self, x: np.ndarray) -> float:
        """
        Returns the density at any given point x.
        :param x:
        :return:
        """
        if self.data is None:
            raise ValueError("Model hasn't been fit yet!")
        if self.data.shape[1] != x.shape[0]:
            raise ValueError(
                "Data on which the model has been fit is not of the same dimensions as the data passed to predict!")

        kernel_values = self.kernel(np.linalg.norm((self.data - x) / self.bandwidth, axis=1))
        density = np.sum(kernel_values) / (self.data.shape[0] * self.bandwidth ** self.data.shape[1])
        return density

    def visualize2D(self, n_points_grid: int = 100) -> None:
        """
        Helps visualize the predicted KDE in 2D.
        :param n_points_grid: The side length of the square grid.
        :return:
        """
        if self.data is None:
            raise ValueError("Model hasn't been fit yet!")
        if self.data.shape[1] != 2:
            raise ValueError("Can only visualise 2D data.")

        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, n_points_grid),
            np.linspace(y_min, y_max, n_points_grid)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        z = np.array([self.predict(x) for x in grid_points]).reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, z, levels=20, cmap='viridis')
        plt.colorbar(label='Density')
        plt.scatter(self.data[:, 0], self.data[:, 1], c='red', alpha=0.5, s=20, label='Data')
        plt.title(f'KDE, kernel={self.kernel_type}, bandwidth h={self.bandwidth}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()