import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    X = np.array(points)
    x = np.array(query_point)
    
    dist = np.sum(np.power(X-x, 2), axis=1)
    # index = np.arange(X.shape[0])
    # np.column_stack((dist, index))
    sorted_indices = np.argsort(dist)
    result =  X[sorted_indices][:k]
    return [tuple(row) for row in result.tolist()]



