import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# # Take data from a file
# def get_data(filepath):
#     # start_time = time.time()
#     df = pd.read_csv(filepath)
#     # df.head()
#     return  df


# Calculating cosine similarity
def cosine_similarity(x, y):
    if (np.linalg.norm(x) * np.linalg.norm(y)) == 0:
        return 0.0
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# K-means Clustering
def k_means(n_clusters, df):
    # Number of clusters
    k = n_clusters

    # Randomly initializing k points
    init_points = (df.sample(n=k))

    mean_points = []
    for i in init_points.values:
        mean_points.append(i[1:11])

    # Running K-means 20 times
    final_clusters = [[] for j in range(k)]
    y = [[] for j in range(k)]
    for i in range(20):
        clusters = [[] for j in range(k)]
        for point in df.values:
            distances = [cosine_similarity(point[1:11], mean) for mean in mean_points]
            nearest_mean_index = np.argmax(distances)
            clusters[nearest_mean_index].append(point)
            y[nearest_mean_index].append(point[1:11])

        for j in range(k):
            if len(clusters[j]) > 0:
                mean_points[j]=np.mean(y[j], axis=0)
        final_clusters = clusters
    return final_clusters


# Calculation of Silhouette Coefficient for each sample
def silhouette_coefficient(n_clusters, final_clusters, df):
    s = 0
    k = n_clusters
    silhouette_coeff = 0
    for i in range(len(df)):
        cluster_index = -1
        for j in range(k):
            if tuple(df.values[i]) in [tuple(x) for x in final_clusters[j]]:
                cluster_index = j
                break
        if cluster_index == -1:
            continue

        # Calculating a
        a = 0
        dist_a = [cosine_similarity(df.values[i][1:11], other_pts[1:11]) for other_pts in final_clusters[cluster_index] if
                     not np.array_equal(df.values[i], other_pts)]
        if len(dist_a) > 0:
            a = np.mean(dist_a)

        # Calculating b
        b = np.inf
        dist_b = []
        for j in range(k):
            if j == cluster_index:
                continue
            for other_pts in final_clusters[j]:
                dist = cosine_similarity(df.values[i][1:11], other_pts[1:11])
                dist_b.append(dist)
            if len(dist_b) > 0:
                if np.mean(dist_b) < b:
                    b = np.mean(dist_b)
        if a != 0 and b != np.inf:
            s = (b - a) / max(a, b)

        silhouette_coeff += s
    silhouette_coeff /= len(df.values)
    return silhouette_coeff


def kmeans_to_file(n_clusters, final_clusters, filename):
    f = open(filename, "w")
    for j in range(n_clusters):
        user_count = 0
        for point in final_clusters[j]:
            x = point[0]
            x = x[5:]
            if user_count == 0:
                f.write(x)
            else:
                f.write(", ")
                f.write(x)
            user_count += 1
        if j < n_clusters-1:
            f.write("\n")
    f.close()


def agglomerative_clustering(n_clusters, df):
    # Single Linkage Agglomerative Hierarchical Clustering
    k = n_clusters
    cosine_distances = squareform(pdist(df.values[:, 1:11], metric='cosine'))

    clusters = [[i] for i in range(len(df.values))]
    while len(clusters) > k:
        min_dist = np.inf
        merge_cluster_index1 = -1
        merge_cluster_index2 = -1
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = np.min(cosine_distances[np.ix_(np.ravel(clusters[i]),np.ravel(clusters[j]))])
                if dist < min_dist:
                    min_dist = dist
                    merge_cluster_index1 = i
                    merge_cluster_index2 = j
        clusters[merge_cluster_index1].extend(clusters[merge_cluster_index2])
        del clusters[merge_cluster_index2]
    return clusters


def agglomerative_to_file(n_clusters, clusters, filename):
    f = open(filename, "w")
    for j in range(n_clusters):
        user_count = 0
        for point in clusters[j]:
            x = point + 1
            if user_count == 0:
                f.write(str(x))
            else:
                x = point + 1
                f.write(", ")
                f.write(str(x))
            user_count += 1
        if j < n_clusters-1:
            f.write("\n")
    f.close()


def jaccard_coefficients(file1, file2, n_clusters):
    # Reading data from kmeans.txt and storing it in result1
    with open(file1, "r") as f:
        lines = [line.strip().split(",") for line in f.readlines()]
    for i in range(len(lines)):
        if lines[i][0] == '':
            lines[i][0] = '981'
    result1 = [[int(x) for x in line] for line in lines]

    # Reading data from agglomerative.txt and storing it in result1
    with open(file2, "r") as f:
        lines = [line.strip().split(",") for line in f.readlines()]
    for i in range(len(lines)):
        if lines[i][0] == '':
            lines[i][0] = '981'
    result2 = [[int(x) for x in line] for line in lines]

    # Calculating Jaccard Similarity for each corresponding cluster
    jaccard_list = []
    for i in range(n_clusters):
        jaccard_similarities = []
        A = set(result1[i])
        for j in range(n_clusters):
            B = set(result2[j])
            intersection = len(A.intersection(B))
            union = len(A.union(B))
            jaccard_similarity = intersection / union
            jaccard_similarities.append(jaccard_similarity)
        jaccard_list.append(jaccard_similarities)
    return jaccard_list


if __name__ == "__main__":
    df = pd.read_csv('travel.csv')
    clusters_dict = {}
    max_silhouette = -2
    optim_clusters = -1
    for i in range(3, 7):
        clusters_dict[i] = k_means(n_clusters=i, df=df)
        silhouette_value = silhouette_coefficient(n_clusters=i, final_clusters=clusters_dict[i], df=df)
        print(f'k={i}')
        print(f'The Silhouette Coefficient is : {silhouette_value}')
        if silhouette_value > max_silhouette:
            max_silhouette = silhouette_value
            optim_clusters = i
    print(f'The optimal value is :\nk = {optim_clusters}')
    kmeans_to_file(optim_clusters,clusters_dict[optim_clusters],"kmeans.txt")
    agglo_clusters = agglomerative_clustering(optim_clusters,df)
    agglomerative_to_file(optim_clusters,agglo_clusters,"agglomerative.txt")
    jaccard_matrix = jaccard_coefficients("kmeans.txt", "agglomerative.txt", optim_clusters)
    for i in range(len(jaccard_matrix)):
        print(f'jaccard_similarities for cluster {i} : {jaccard_matrix[i]}')





