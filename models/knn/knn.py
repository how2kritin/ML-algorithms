"""
Implements a generic KNN (K-Nearest Neighbours) model, with fit and predict functionality.
"""
# References:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# https://pythonspeed.com/articles/pandas-vectorization/
from collections import Counter
from typing import Callable
from pandas import DataFrame, Series
import numpy as np
import json


class KNN:
    def __init__(self, k: int, distance_metric_function: Callable):
        """
        :param k: The number of nearest neighbours to be considered.
        :param distance_metric_function: The function to be used to calculate distances. Must take 2 vectors and return a float.
        """
        self.k: int = k
        self.distance: Callable = distance_metric_function
        self.X_train: DataFrame = None
        self.y_train: DataFrame = None

    def fit(self, X: DataFrame, y: DataFrame) -> None:
        self.X_train = X
        self.y_train = y

    def non_vectorised_predict(self, X: DataFrame, predicted_column_name: str) -> Series:
        """
        Returns class labels for each data sample in the test set, by comparing it with its K Nearest Neighbours in the
        training set.

        To determine the class label, I shall use the most common label of the K Nearest Neighbours as my voting method.

        In the voting method, the tie-breaker is first by distance, and next by an index ordering of points in the
        training set.

        :param X: The input dataset.
        :param predicted_column_name: The name of the output column that was predicted, i.e., the 'y' output column's name to the 'X' input.
        :return:
        """
        if self.X_train.empty or self.y_train.empty:
            raise Exception("Cannot predict before training model!")

        y_pred = []

        for _, entry in X.iterrows():
            distances_to_points = list()

            for i in range(len(self.X_train)):
                distance = self.distance(entry, self.X_train.iloc[i])
                distances_to_points.append((distance, i))

            distances_to_points.sort()

            label_count_map = dict()
            most_common_label: str = ""
            most_common_label_count: int = 0

            for i in range(self.k):  # choose the k nearest neighbors
                neighbour_idx = distances_to_points[i][1]
                label = self.y_train.iloc[neighbour_idx]

                if label in label_count_map:
                    label_count_map[label] += 1
                else:
                    label_count_map[label] = 1

                if label_count_map[label] > most_common_label_count:
                    # tie breaking by order of labels, i.e., sorted order of distances.
                    most_common_label = label
                    most_common_label_count = label_count_map[label]

            y_pred.append(most_common_label)

        return Series(y_pred, name=predicted_column_name)

    def predict(self, X: DataFrame, predicted_column_name: str = 'predictions') -> Series:
        """
        Reference: https://pythonspeed.com/articles/pandas-vectorization/

        Vectorized distance finding operation between each row in test set and training set.

        Returns class labels for each data sample in the test set, by comparing it with its K Nearest Neighbours in the
        training set.

        To determine the class label, I shall use the most common label of the K Nearest Neighbours as my voting method.

        In the voting method, the tie-breaker is first by distance, and next by an index ordering of points in the
        training set.

        :param X: The input dataset.
        :param predicted_column_name: The name of the output column that was predicted, i.e., the 'y' output column's name to the 'X' input. "predictions" by default.
        :return:
        """
        if self.X_train.empty or self.y_train.empty:
            raise Exception("Cannot predict before training model!")

        # Figured out that directly using the underlying numpy ndarray is faster than handling pandas DataFrame.
        # This is a better way to vectorize. Noticed upto 9x speedup this way.
        X_train = self.X_train.to_numpy()
        y_train = self.y_train.to_numpy()
        X = X.to_numpy()

        y_pred = []

        for entry in X:
            distances = self.distance(entry, X_train)
            k_nearest_indices = np.argpartition(distances, self.k)[:self.k]
            nearest_labels = y_train[k_nearest_indices]
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]  # faster than looping and counting.
            y_pred.append(most_common_label)

        return Series(y_pred, name=predicted_column_name)

    def tune_hyperparameters(self, X_validation: DataFrame, y_validation: Series, distance_metrics: list[Callable],
                             observations_output_file_path: str, k_upper_bound: int) -> list[tuple]:
        """
        Automatically tunes the model to the best hyperparameters (k and distance), given a list of possible distance metrics.
        The best hyperparameters are chosen solely on the basis of accuracy.
        Typically, we tune k until sqrt(n), where n is the total number of records in the training set.
        :param X_validation:
        :param y_validation:
        :param distance_metrics: A list of distance functions.
        :param observations_output_file_path: Path to file where the observations are to be stored in JSON format.
        :param k_upper_bound: Upper bound for the values of k to be tested. Usually supposed to be sqrt(num training dataset entries).
        :return:
        """
        if self.X_train.empty or self.y_train.empty:
            raise Exception("Cannot tune hyperparameters before training model!")

        k_dist_accuracy_observations = []
        curr_best_k = self.k
        curr_best_distance_metric = self.distance
        curr_best_accuracy = 0

        for k in range(1, k_upper_bound + 1):
            for metric_idx in range(len(distance_metrics)):
                self.k = k
                self.distance = distance_metrics[metric_idx]

                y_pred = self.predict(X_validation)
                accuracy = np.mean(y_validation == y_pred)

                observation = {'k': k, 'distance_metric_idx': metric_idx, 'accuracy': accuracy}
                k_dist_accuracy_observations.append(observation)
                with open(observations_output_file_path, 'a') as file:
                    file.write(json.dumps(observation) + '\n')

                if accuracy > curr_best_accuracy:
                    print(f"New best accuracy: {accuracy} at k = {k} and dist_idx = {metric_idx}")
                    curr_best_k = self.k
                    curr_best_distance_metric = self.distance
                    curr_best_accuracy = accuracy

        # set this model's hyperparams to the best hyperparams that were determined.
        print("======================================")
        print(
            f"Final best accuracy: {curr_best_accuracy} at k = {curr_best_k} and dist_metric_idx = {distance_metrics.index(curr_best_distance_metric)}")
        self.k = curr_best_k
        self.distance = curr_best_distance_metric

        return k_dist_accuracy_observations
