import math
from functools import cache
import pandas as pd
import time
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# !! This code is basically the same as held_kerp_b_kmeans_planning_test.ipynb. Please see the test result over there and check if it works well. !!

# -----------------------------------------------------------------------------
# Algorithms
# -----------------------------------------------------------------------------

def held_karp(adj_matrix, is_closed=False):
    """
    Solve the traveling salesman problem using Held-Karp algorithm.
    
    Parameters:
    adj_matrix: adjacency matrix where adj_matrix[i][j] is the distance from node i to node j
    is_closed: if true, the path will be hamiltonian circuit, if false the path will start at node 0 and end at node n-1
    
    Returns:
    path: list representing the shortest path from the first node to the last node
    distance: the shortest distance for the path
    """
    
    n = len(adj_matrix)
    if n == 0:
        return [], 0
    
    # Notes on the bit-mask representation (node index starts from 0):
    # S = 1 << (i - 1) <=> node i in S
    # ex. S = 0b101 <=> node 1 and 3 in S
    # ex. S = 0b1000 <=> node 4 in S
    # ex. S = 0b0 <=> node 0 in S

    @cache
    def g(S, k):
        """
        Recursive function to find the minimum distance to reach node k from node 0 after visiting all nodes in set S.
        
        Parameters:
        S: bit-mask representing the set of nodes, where 0, k not in S.
        k: the last node to visit
        
        Returns:
        dist: the minimum distance to reach node k from node 0 through all nodes in S
        prev: the previous node before reaching k
        """
        if S == 0:
            return adj_matrix[0][k], 0
        
        # Evaluate minimum distance by checking all possible previous nodes in the set S
        return min(
            (g(S ^ (1 << (i - 1)), i)[0] + adj_matrix[i][k], i)  # dist(S - {i}, i) + dist(i, k)
            for i in range(1, n) if S & (1 << (i - 1))  # For all i in S
        )

    if is_closed:
        # Initialize S as all nodes except node 0, S = 0b111...1 (n-1 bits)
        S = (1 << (n - 1)) - 1
        # Solve the TSP for the closed path
        dist, last = min(
            (g(S ^ (1 << (i - 1)), i)[0] + adj_matrix[i][0], i)  # dist(S= {i}, i) + dist(i, 0)
            for i in range(1, n)
        )
        # Reconstruct the path by backtracking from the last node
        path = []
        while S:
            path.append(last)
            S ^= 1 << (last - 1)  # S = S - {last}
            _, last = g(S, last)
        path = [0] + path[::-1] + [0]
    else:
        # Initialize S as all nodes except node 0 and node n-1, S = 0b011...1 (n-1 bits with MSB = 0)
        S = (1 << (n - 2)) - 1
        # Solve the TSP for the open path
        dist, last = g(S, n - 1)
        # Reconstruct the path by backtracking from the last node
        path = []
        while S:
            path.append(last)
            S ^= 1 << (last - 1)
            _, last = g(S, last)
        path = [0] + path[::-1] + [n - 1]
       
    # Return the path, distance
    return path, dist


# Points to adj matrix
def points_to_adj_matrix(points):
    n = len(points)
    adj_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            adj_matrix[i][j] = adj_matrix[j][i] = math.dist(points[i], points[j])
    return adj_matrix


def balanced_kmeans(X, k, max_iter=100, tol=1e-4):
    """
    Perform balanced k-means clustering on the given data.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
    - k: number of clusters
    - max_iter: maximum number of iterations
    - tol: tolerance for convergence

    Returns:
    - labels: array of shape (n_samples,) with cluster assignments
    - centroids: array of shape (k, n_features)
    """
    n_samples, n_features = X.shape

    # Determine the cluster sizes
    base_size = n_samples // k
    remainder = n_samples % k
    cluster_sizes = [base_size + 1 if i < remainder else base_size for i in range(k)]

    # Initialize centroids randomly from data points
    rng = np.random.default_rng()
    centroids = X[rng.choice(n_samples, k, replace=False)]
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        # Expand centroids to match slots
        centroids_expanded = np.vstack([
            np.tile(centroids[i], (cluster_sizes[i], 1))
            for i in range(k)
        ])

        # Compute cost matrix (squared Euclidean distances)
        cost_matrix = np.linalg.norm(
            X[:, np.newaxis] - centroids_expanded[np.newaxis, :],
            axis=2
        ) ** 2

        # Solve the assignment problem using the Hungarian algorithm
        # cf. https://en.wikipedia.org/wiki/Assignment_problem
        _, col_ind = linear_sum_assignment(cost_matrix)

        # Map assignments to cluster labels
        slot_to_cluster = np.hstack([
            np.full(size, cluster_idx)
            for cluster_idx, size in enumerate(cluster_sizes)
        ])
        new_labels = slot_to_cluster[col_ind]

        # Update centroids
        new_centroids = np.array([
            X[new_labels == c].mean(axis=0) if np.any(new_labels == c)
            else centroids[c]
            for c in range(k)
        ])

        # Check for convergence
        shifts = np.linalg.norm(new_centroids - centroids, axis=1)
        if np.all(shifts < tol):
            centroids = new_centroids
            labels = new_labels
            break

        centroids = new_centroids
        labels = new_labels

    return labels, centroids


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

# File loading will take some time, so it is better to load the file only once
poi_coords = pd.read_csv('./planning_models/tsp/poi_coords.csv')


# -----------------------------------------------------------------------------
# Planning Function
# -----------------------------------------------------------------------------

def planning(place_infos, n_clusters):
    """
    Planning with the first location as a fixed start and end point (accomodation)
    
    Parameters:
    place_infos: list of POI IDs and their coordinates
    n_clusters: number of clusters for balanced K-means
    
    Returns:
    paths: list of paths for each cluster starting and ending with accomodation
    """
    # Separate accomodation and other points
    accomo_id = str(place_infos[0][0])
    accomo_coords = [place_infos[0][1], place_infos[0][2]]
    
    other_place_infos = place_infos[1:]
    poi_ids = [str(place[0]) for place in other_place_infos]
    points = [[place[1], place[2]] for place in other_place_infos]
    
    # Perform balanced K-means clustering
    labels, _ = balanced_kmeans(points, n_clusters)

    # Prepare the adjacency matrix for each cluster
    paths = []
    for c in range(n_clusters):
        mask = (labels == c)
        cluster_points = np.array(points)[mask]
        cluster_ids = np.array(poi_ids)[mask]
        
        # Include 숙소 in the cluster points for TSP calculation
        cluster_points_with_숙소 = np.vstack([accomo_coords, cluster_points])
        adj_matrix = points_to_adj_matrix(cluster_points_with_숙소)
        
        # Solve TSP (with 숙소 at the start and end)
        path, _ = held_karp(adj_matrix, is_closed=True)
        
        # Map path indices to actual POI IDs (starting and ending with 숙소)
        ordered_place_ids = [accomo_id] + cluster_ids[path[1:-1] - 1].tolist() + [숙소_id]
        paths.append(ordered_place_ids)
    
    return paths
