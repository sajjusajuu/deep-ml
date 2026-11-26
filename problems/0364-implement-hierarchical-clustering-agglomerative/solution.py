import numpy as np

def agglomerative_clustering(X: list, n_clusters: int, linkage: str = 'single') -> list:
    """
    Perform agglomerative hierarchical clustering.
    
    Args:
        X: Data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        linkage: 'single', 'complete', or 'average'
    
    Returns:
        List of cluster labels for each sample
    """
    m = len(X)
    d = len(X[0])
    x = np.array(X)

    distance_mat = np.zeros((m,m))
    labels = np.arange(m)
    lc = m

    for i in range(m):
        for j in range(i):
            distance_mat[i][j] = np.sqrt(np.sum(np.power(x[i] - x[j], 2)))
    

    while True:
        if lc == n_clusters:
            break

        min_i = -1
        min_j = -1
        min_val = -1
        for i in range(m):
            if labels[i] != i:
                continue
            for j in range(i):
                if labels[j] != j:
                    continue
                if min_val < 0 or distance_mat[i][j] < distance_mat[min_i][min_j]:
                    min_i = i
                    min_j = j
                    min_val = distance_mat[i][j]
    
        lc -= 1
        labels[min_i] = min_j
    
        #merging min_i -> min_j && min_i > min_j
        for i in range(m):
            if min_j > i:
                if linkage == 'single':
                    distance_mat[min_j][i] = min(distance_mat[min_j][i], distance_mat[min_i][j])
                elif linkage == 'complete':
                    distance_mat[min_j][i] = max(distance_mat[min_j][i], distance_mat[min_i][j])
                elif linkage == 'average':
                    distance_mat[min_j][i] = (distance_mat[min_j][i] + distance_mat[min_i][j])/2
        for i in range(m):
            if i > min_j: 
                if linkage == 'single':
                    distance_mat[i][min_j] = min(distance_mat[i][min_j], distance_mat[i][min_i])
                elif linkage == 'complete':
                    distance_mat[i][min_j] = max(distance_mat[i][min_j], distance_mat[i][min_i])
                elif linkage == 'average':
                    distance_mat[i][min_j] = (distance_mat[i][min_j] + distance_mat[i][min_i])/2
        
    new_labels = np.arange(m)
    new_labels[0] = 0
    current_label = 0
    for i in range(1, m):
        e = i
        while labels[e] != e:
            e = labels[e]
        if e == i:
            new_labels[i] = current_label + 1
            current_label += 1
        else:
            new_labels[i] = new_labels[e]


    return new_labels.tolist()