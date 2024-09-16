import math
from functools import cache
import pandas as pd
import time

# -----------------------------------------------------------------------------
# Algorithm: Held-Karp Algorithm (Dynamic Programming)
# -----------------------------------------------------------------------------

def held_karp_solver(adj_matrix):
    """
    Solve the (modified) traveling salesman problem using Held-Karp algorithm (Dynamic Programming).
    
    Parameters:
    adj_matrix: adjacency matrix where adj_matrix[i][j] is the distance from node i to node j
    
    Returns:
    path: list representing the shortest path from the first node to the last node
    distance: the shortest distance for the path
    """
    
    n = len(adj_matrix)
    if n == 0:
        return [], 0
    
    @cache
    def g(S, k):
        """
        Recursive function to find the minimum distance to reach node k from node 0 after visiting all nodes in set S.
        
        Parameters:
        S: bit-mask representing the set of nodes to visit in-between (ex. node 1, 3 = 0b101)
        k: the last node to visit
        
        Returns:
        dist: the minimum distance to reach node k from node 0 through all nodes in S
        prev: the previous node before reaching k
        """
        if S == 0:
            return adj_matrix[0][k], 0
        
        # Evaluate minimum distance by checking all possible previous nodes in the set S
        return min(
            (g(S ^ (1 << (i - 1)), i)[0] + adj_matrix[i][k], i) # dist(S - {i}, i) + dist(i, k)
            for i in range(1, n) if S & (1 << (i - 1)) # For all i in S
        )

    # Initialize S as all nodes except node 0 and node n-1
    S = (1 << (n - 1)) - 1 
    dist, prev = g(S, n - 1)

    # Reconstruct the path by backtracking from the last node
    path = []
    while S:
        path.append(prev)
        S ^= 1 << (prev - 1) # S = S - {prev}
        _, prev = g(S, prev)

    return [0] + path[::-1], dist

# Points to adj matrix
def points_to_adj_matrix(points):
    n = len(points)
    adj_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            adj_matrix[i][j] = adj_matrix[j][i] = math.dist(points[i], points[j])
    return adj_matrix

# -----------------------------------------------------------------------------
# Planning Function
# -----------------------------------------------------------------------------

# File loading will take some time, so it is better to load the file only once
poi_coords = pd.read_csv('TSP/poi_coords.csv')

def planning(poi_ids):
    """
    Plan the shortest path to visit all POIs using Held-Karp algorithm.
    
    Parameters:
    poi_ids: list of POI IDs to visit, first element is the starting point and last element is the ending point
    
    Returns:
    path: list representing the shortest path to visit all POIs

    Note:
    - The number of POIs to visit should be less than 15 due to the exponential time complexity of the algorithm (~ 0.5 sec for 15 POIs)
    """

    if(len(poi_ids) > 15):
        print("Too many POIs to visit (>15)")
        return None
    
    poi_indexes = poi_coords['POI_ID'].searchsorted(poi_ids)
    points = poi_coords.iloc[poi_indexes][['X_COORD', 'Y_COORD']].values
    
    adj_matrix = points_to_adj_matrix(points)
    path, _ = held_karp_solver(adj_matrix)
    return [poi_ids[i] for i in path]

# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

# Randomly select 5 POIs to visit
poi_indexes = [123, 2341, 345, 234, 1234]
poi_ids = poi_coords['POI_ID'][poi_indexes].tolist()
print(poi_ids)
start = time.time()
path = planning(poi_ids)
end = time.time()
print(path)
print(f"Time taken: {end-start:.4f})")
