import math 

def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    d = len(a)
    dis = 0
    for i in range(d):
        dis += math.pow(a[i] - b[i], 2)
    return math.sqrt(dis)


def mean(points: list[tuple[float, ...]]) -> tuple[float, ...]:
    d = len(points[0])
    mean = [0.0 for i in range(d)]
    for i in range(d):
        for point in points:
            mean[i] += float(point[i])

    for i in range(d):
        mean[i] = float(mean[i]/len(points))
    
    return tuple(mean)

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:

    for i in range(max_iterations):
        grouping_by_points = [ [] for i in range(k)]
        for point in points:
            closest_centroid = 0
            closest_centroid_dist =  distance(point, initial_centroids[closest_centroid])
            for i in range(1,k):
                centroid = initial_centroids[i]
                dist = distance(point, centroid)
                if dist < closest_centroid_dist:
                    closest_centroid = i
                    closest_centroid_dist = dist
            grouping_by_points[closest_centroid].append(point)
        for i in range(k):
            initial_centroids[i] = mean(grouping_by_points[i])
        
    return initial_centroids