"""
A K-Means algorithm implementation. I will be using K-Means++ to initialise my centroids.
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""
from pandas import DataFrame
import numpy as np
from typing import Self


class K_Means:
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 1e-4, random_state: int = None):
        """
        :param n_clusters: Number of clusters. 8 by default (same as KMeans from Scikit-learn).
        :param max_iter: Maximum number of iterations of the k-means algorithm for a single run. 300 by default.
        :param tol: Tolerance; to declare convergence. 1e-4 by default.
        :param random_state: Random state to make the random sampling deterministic. Not set by default.
        """
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.random_state: int = random_state

        self.centroids: list = []

    def _euclidean_distance_fully_vectorized(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    def init_kmeans_plus_plus(self, X: DataFrame) -> Self:
        """
        Uses K-Means++ to initialise centroids.
        Ref: https://en.wikipedia.org/wiki/K-means%2B%2B
        :param X: A pandas dataframe with numeric columns, each representing a feature.
        :return: The object itself.
        """
        self.centroids = []
        X_vals = X.values  # converting dataframe to ndarray

        # K-Means++
        # Step-1: choose a center uniformly at random among the data points.
        self.centroids.append(X.sample(n=1, random_state=self.random_state).values.flatten())

        # initializing distances to infinity
        distances = np.full(X_vals.shape[0], np.inf)

        # iterating to get k-1 more cluster centers;
        for _ in range(self.n_clusters - 1):
            # Step-2: distances between each point and the currently chosen cluster centers:
            new_distances = self._euclidean_distance_fully_vectorized(X_vals, np.array([self.centroids[-1]]))[:, 0]
            distances = np.minimum(distances, new_distances)

            # now, the distances array has the min distance from any point in our centroid array to the dataset points.
            probabilities = distances ** 2 / np.sum(distances ** 2)

            # Step-3: choose a new centroid with probability proportional to distance squared.
            self.centroids.append(
                X.sample(n=1, weights=probabilities, random_state=self.random_state).values.flatten())

        self.centroids = np.array(self.centroids)
        return self

    def fit(self, X: DataFrame) -> Self:
        self.init_kmeans_plus_plus(X)

        # now, iterate until convergence of centroids:
        for _ in range(self.max_iter):
            # find the closest centroid to each point.
            distance_matrix = self._euclidean_distance_fully_vectorized(X.values, np.array(self.centroids))
            closest_centroids = np.argmin(distance_matrix, axis=1)

            # update the centroids now; basically move each centroid to the mean of points assigned to it currently.
            new_centroids = []
            for i in range(self.n_clusters):
                points_assigned_to_centroid = X.values[closest_centroids == i]
                if len(points_assigned_to_centroid) > 0:
                    new_centroids.append(points_assigned_to_centroid.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[
                                             i])  # if no points were assigned to this centroid, then simply keep the old centroid.

            new_centroids = np.array(new_centroids)

            # if the new centroids are close to the old centroids, then convergence has been reached. Terminate the loop here.
            if np.linalg.norm(self.centroids - new_centroids) <= self.tol:  # sum of distances difference between each centroid must be within tolerance.
                break

            self.centroids = new_centroids

        return self

    def getCentroids(self) -> np.ndarray:
        if len(self.centroids) == 0:
            raise Exception("The model hasn't been fit yet!")

        return self.centroids

    def predict(self, X: DataFrame) -> np.ndarray:
        """
        Returns the cluster number of each data point. This corresponds to the index of the centroids present in 'self.centroids' of the fit model.
        :param X:
        :return:
        """
        distance_matrix = self._euclidean_distance_fully_vectorized(X.values, np.array(self.centroids))
        closest_centroids = np.argmin(distance_matrix, axis=1)
        return closest_centroids

    def getCost(self, X: DataFrame) -> float:
        """
        Returns the Within-Cluster Sum of Squares (WCSS) cost for the current centroids.
        :param X:
        :return:
        """
        X_vals = X.values  # converting dataframe to ndarray
        cost = 0

        distance_matrix = self._euclidean_distance_fully_vectorized(X_vals, np.array(self.centroids))
        closest_centroids = np.argmin(distance_matrix, axis=1)

        for i in range(self.n_clusters):
            points_assigned_to_centroid = X_vals[closest_centroids == i]
            cost += np.sum((points_assigned_to_centroid - self.centroids[i]) ** 2)

        return cost
