import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, dims: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            dims: int
                the dimension the clusters will be in.
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if k < 1 or type(k) != int:
            raise ValueError("k must be an integer > 0")
        if dims < 1 or type(dims) != int:
            raise ValueError("dims must be an integer > 0")
        if tol < 0:
            raise ValueError("minimum tolerance must be > 0")
        if max_iter < 1 or type(max_iter) != int:
            raise ValueError("iteration number must be an integer > 0")

        self.max_iter = max_iter
        self.k = k
        self.dims = dims
        self.tol = tol
        self.error = None # to be calculated once fit occurs.
        self.centroids = np.array([])

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # Quality check the inputs.
        if len(mat) < self.k:
            raise ValueError("Input matrix doesn't have enough values for "
                             f"the provided k value of {self.k}.")
        if len(mat[0]) != self.dims:
            raise ValueError("Input matrix does not have the correct "
                             f"number of dimensions {self.dims}.")
        if type(mat[0][0]) != np.float64:
            raise ValueError("Input matrix does not contain floats, contains "
                             f"{type(mat[0][0])}.")

        # Randomly assign centroids to points if not already assigned.
        if len(self.centroids) == 0:
            centroid_indeces = np.random.choice(
                list(range(len(mat))), self.k, replace=False
            )
            self.centroids = np.array(
                list(map(lambda i: mat[i], centroid_indeces))
            )

        max_change = np.inf
        # Perform the fitting, stopping at max iterations or when change < tol.
        for _ in range(self.max_iter):
            predictions = self.predict(mat)
            max_change = self._update_centroids_and_return_distance(
                mat, predictions
            )
            # Exit early if the centroids have stabilized.
            if max_change < self.tol:
                break
        
        # Calculate overall error using distances.
        self.error = 0
        for index in range(len(mat)):
            obs = mat[index]
            centroid = self.centroids[predictions[index]]
            self.error += self._get_distance(obs, centroid)**2 / len(mat)
        return
        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        if len(self.centroids) == 0:
            raise ValueError("The model cannot predict before being trained.")

        # Store predictions in a list before numpying it at the end.
        predictions = []
        for observation in mat:
            # Start with dummy value to be overriden almost immediately.
            closest_centroid = (np.inf, np.array([]))
            for centroid_index in range(len(self.centroids)):
                # Calculate distance and compare to current best.
                distance = self._get_distance(observation, self.centroids[centroid_index])
                if distance < closest_centroid[0]:
                    closest_centroid = (distance, centroid_index)
            predictions.append(closest_centroid[1])
        return np.array(predictions)

    
    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids

    def _get_distance(self, observation: np.ndarray, centroid: np.ndarray):
        """Calculate distance between two points in n-dimensional space."""
        square_diffs = list(map(
            lambda i: (observation[i] - centroid[i])**2, range(len(observation))
        ))
        return np.sqrt(sum(square_diffs))

    def _update_centroids_and_return_distance(self, mat: np.ndarray,
                                              predictions: np.ndarray):
        """
        From predictions, update centroid locations. At the same time, keep
        track of what the max distance change between any centroid and its
        updated value is.

        inputs:
            mat (array): the observation*dimension matrix.
            predictions (array): 1D array of cluster predictions for each point
        outputs:
            the max change between any centroid and its successor. Also updates
            the centroid object variable.

        """

        clusters = {}
        for index in range(len(predictions)):
            observation = mat[index].copy()
            label = predictions[index]
            # Populate dictionary of cluster labels with summed values in each
            # dimension and the number of points in the cluster for averaging.
            if label in clusters:
                clusters[label][0] += observation
                clusters[label][1] += 1
            else:
                clusters[label] = [observation, 1]

        # Average the points, calculate max distance change.
        max_dist = 0
        for index in range(len(self.centroids)):
            new_point = clusters[index][0] / clusters[index][1]
            max_dist = max(
                max_dist, self._get_distance(new_point, self.centroids[index])
            )
            self.centroids[index] = new_point
        return max_dist