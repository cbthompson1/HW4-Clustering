import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if len(X) != len(y):
            raise ValueError("Mismatch in observations and labels.")

        silhouette = []
        for index in range(len(X)):
            # Offshore local computation to private helper functions.
            intra_score = self._intra_score(X, y, index)
            inter_score = self._inter_score(X, y, index)
            # Calculate silhouette score and add to return list.
            silhouette_score = (inter_score - intra_score) / max(inter_score, intra_score)
            silhouette.append(silhouette_score)
        return np.array(silhouette)

    def _intra_score(self, observations, labels, index):
        """Determine intra score for provided index."""
        cluster_label = labels[index]
        observation = observations[index].reshape(1,-1)
        # Get all distances of the same cluster.
        intra_distances = cdist(
            observation, observations[labels == cluster_label], "euclidean"
        )[0]
        # Return average (length - 1 to account for self-comparison).
        distance_sums = sum(intra_distances[intra_distances != 0])
        return distance_sums / (len(intra_distances) - 1)

    def _inter_score(self, observations, labels, index):
        """Determine inter score for provided index."""
        intra_cluster = labels[index]
        observation = observations[index].reshape(1,-1)

        # Iterate on all clusters (minus the one the observation is in).
        unique_clusters = set(labels)
        unique_clusters.remove(intra_cluster)
        # While iterating through clusters, compare against current best.
        min_cluster_dist = np.inf
        for cluster in unique_clusters:
            inter_distances = cdist(
                observation, observations[labels == cluster], "euclidean"
            )
            min_cluster_dist = min(
                min_cluster_dist, np.mean(inter_distances)
            )
        return min_cluster_dist
