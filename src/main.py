import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import combinations

def generate(points_count):
    # Generating initial 20 unique points
    num_initial_points = 20
    points = np.random.uniform(low=-5000, high=5000, size=(0, 2))

    while len(np.unique(points, axis=0)) < num_initial_points:
        new_point = np.random.uniform(low=-5000, high=5000, size=(1, 2))

        if not np.any(np.all(np.isclose(points, new_point), axis=1)):
            points = np.vstack([points, new_point])

    # Creating an additional 20,000 points considering the given conditions
    for _ in range(points_count):
        # Randomly choosing one of the generated points
        selected_point = points[np.random.randint(0, len(points))]

        # Determining distances to edges
        distance_to_edge_x = 5000 - abs(selected_point[0])
        distance_to_edge_y = 5000 - abs(selected_point[1])

        while True:
            # Adjusting the interval if the point is close to the edge
            if 100 > distance_to_edge_x > -100:
                x_offset = np.random.uniform(-distance_to_edge_x, distance_to_edge_x)
            else:
                x_offset = np.random.uniform(-100, 100)
            if 100 > distance_to_edge_y > -100:
                y_offset = np.random.uniform(-distance_to_edge_y, distance_to_edge_y)
            else:
                y_offset = np.random.uniform(-100, 100)

            # Adding a new point in the two-dimensional space considering the offsets
            new_point = selected_point + [x_offset, y_offset]

            # Ensuring the new point stays within the boundaries [-5000, 5000]
            new_point = np.clip(new_point, -5000, 5000)

            # Checking if the point 'new_point' already exists in the 'points' array
            if not np.any(np.all(np.isclose(points, new_point), axis=1)):
                # If not, adding the new point and exiting the loop
                points = np.vstack([points, new_point])
                break

    # Visualizing the points
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5, label='Points')
    plt.title('Generated Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return points

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def merge_clusters(distances, cluster_indices):

    new_row_column = np.maximum(distances[:, cluster_indices[0]], distances[:, cluster_indices[1]])
    new_row_column = np.expand_dims(new_row_column, axis=1)

    # Writing values of the new row into the matrix
    distances[cluster_indices[0], :] = new_row_column.T  # Overwriting the row
    distances[:, cluster_indices[0]] = new_row_column[:, 0]  # Overwriting the column

    # Setting a value of 0 at the intersection of the new row and column
    distances[cluster_indices[0], cluster_indices[0]] = 0

    # Deleting the row and column
    distances = np.delete(distances, cluster_indices[1], axis=0)  # Deleting the row
    distances = np.delete(distances, cluster_indices[1], axis=1)  # Deleting the column

    return distances

# Agglomerative clustering based on the distance matrix
def agglomerative_clustering(points, k):
    n = len(points)

    current_size = n

    # Computing the distance matrix
    distances = cdist(points, points)

    # List to store clusters
    clusters = [[i] for i in range(n)]

    while len(clusters) > k:
        min_dist = np.inf
        merge_indices = (0, 0)

        # Using current_size to limit the matrix size
        for i, j in combinations(range(current_size), 2):
            if distances[i, j] < min_dist:
                min_dist = distances[i, j]
                merge_indices = (i, j)

        clusters[merge_indices[0]] += clusters[merge_indices[1]]
        del clusters[merge_indices[1]]

        # Calling merge_clusters to update distance matrix after merging
        distances = merge_clusters(distances, merge_indices)
        current_size -= 1  # Decreasing the current matrix size

    # Assigning labels based on clusters
    labels = np.zeros(len(points), dtype=int)
    for i, cluster in enumerate(clusters):
        labels[cluster] = i

    print(distances)

    return labels


def cluster_evaluation(points, labels, method):
    for i in np.unique(labels):
        # Extract points belonging to the current cluster
        cluster_points = points[labels == i]

        # Calculate the center based on the specified method
        if method == 'centroid':
            center = np.mean(cluster_points, axis=0)
        elif method == 'medoid':
            center = np.median(cluster_points, axis=0)

        # Calculate distances from the center to each point in the cluster
        distances = [euclidean_distance(center, point) for point in cluster_points]

        # Calculate the average distance from the center to points in the cluster
        avg_distance = np.mean(distances)

        # Check if the average distance meets a certain threshold (500 in this case)
        if avg_distance <= 500:
            print(f"Cluster {i} is successful. Average {method} distance: {avg_distance:.2f}")
        else:
            print(f"Cluster {i} is unsuccessful. Average {method} distance: {avg_distance:.2f}")


def medoid(cluster_points):
    min_sum_distance = np.inf  # Initializing the minimum sum of distances to infinity
    medoid_point = None  # Initializing the medoid point as None

    # Iterating through each point in the cluster
    for point in cluster_points:
        # Calculating the sum of distances from 'point' to all other points in the cluster
        sum_distance = np.sum([euclidean_distance(point, other) for other in cluster_points])

        # Updating the minimum sum of distances and medoid point if a smaller sum is found
        if sum_distance < min_sum_distance:
            min_sum_distance = sum_distance
            medoid_point = point

    return medoid_point  # Returning the point that minimizes the sum of distances (the medoid)


# Function for visualizing clusters
def visualize_clusters(points, labels):
    for i in np.unique(labels):
        # Plotting individual points within each cluster
        cluster_points = points[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20, label=f'Cluster {i}')

    for i in np.unique(labels):
        # Plotting centroids for each cluster
        cluster_points = points[labels == i]
        centroid = np.mean(cluster_points, axis=0)
        plt.scatter(centroid[0], centroid[1], marker='*', color='black', s=50, label=f'Centroid {i}')

    for i in np.unique(labels):
        # Plotting medoids for each cluster
        cluster_points = points[labels == i]
        medoid_point = medoid(cluster_points)  # Assuming there's a 'medoid' function available
        plt.scatter(medoid_point[0], medoid_point[1], marker='*', color='red', s=50, label=f'Medoid {i}')

    plt.title('Agglomerative Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    number_of_points = int(input("Enter the number of points: "))
    points = generate(number_of_points)

    k = int(input("Enter the number of clusters: "))

    # Performing agglomerative clustering
    labels = agglomerative_clustering(points, k)

    # Evaluation based on medoid distance for each cluster
    cluster_evaluation(points, labels, method='medoid')

    # Calculating centroids for each cluster and evaluating them
    cluster_evaluation(points, labels, method='centroid')

    # Visualizing the clusters
    visualize_clusters(points, labels)