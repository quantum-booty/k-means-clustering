import numpy as np
import matplotlib.pyplot as plt


"""
initualise k centroids randomly
assign closest data points to each centroids
move centroids to the mean points of assigned data points
"""


def uniform_init_centroids(k, X):
    n_dims = X.shape[1]
    X_mins = np.min(X, axis=0)
    X_maxs = np.max(X, axis=0)
    return np.random.uniform(X_mins, X_maxs, size=(k, n_dims))


# def init_centroids_pp(k, X, n_init):
#     """kmeans++ initialisation"""
#     n_dims = X.shape[1]
#     X_mins = np.min(X, axis=0)
#     X_maxs = np.max(X, axis=0)

#     cent_init = np.random.randint(np.floor(X_mins), np.ceil(X_maxs), size=(n_init, n_dims))
#     cent_idx = np.arange(n_init)
#     centroids = []
#     chosen = set()

#     for i in range(k):
#         if i == 0:
#             probs = None
#         else:
#             probs = np.array([manhat_dist(centroids[i - 1], centroids[j]) ** 2 for j in cent_idx if i not in chosen])
#             probs = probs / np.sum(probs)

#         chosen_idx = np.random.choice(cent_idx, p=probs)
#         chosen.add(chosen_idx)
#         centroids.append(cent_init[chosen_idx])
#     return centroids


def manhat_dist(x1, x2):
    return np.sum(np.abs(np.atleast_2d(x1 - x2)), axis=1)


def closest_samples(centroids, X):
    assignment = {i: [] for i in range(len(centroids))}
    for j, x in enumerate(X):
        dists = manhat_dist(centroids, x)
        closest_centroid_idx = np.argmin(dists)
        assignment[closest_centroid_idx].append(j)
    return assignment


def predict(centroids, X):
    pred = []
    for x in X:
        dists = manhat_dist(centroids, x)
        closest_centroid_idx = np.argmin(dists)
        pred.append(closest_centroid_idx)
    return np.array(pred)


def adjust_centroids(assignment, X):
    return np.array(
        [
            np.mean(X[indices], axis=0) if indices != [] else np.squeeze(uniform_init_centroids(1, X))
            for indices in assignment.values()
        ]
    )


def train_k_means(centroids_init, X, max_iter):
    centroids = centroids_init
    assignments = None
    for _ in range(max_iter):
        assignments = closest_samples(centroids, X)
        prev_centroids = centroids
        centroids = adjust_centroids(assignments, X)
        if np.all(centroids == prev_centroids):
            break
    assert assignments is not None
    return centroids, assignments


def davis_bouldin_index(centroids, X, assignments):
    DB = 0
    k = len(centroids)
    for i, centroid_i in enumerate(centroids):
        cluster_i = X[assignments[i]]
        sigma_i = np.mean(manhat_dist(cluster_i, centroid_i))
        scores_i = []
        for j, centroid_j in enumerate(centroids):
            if i == j:
                continue
            cluster_j = X[assignments[j]]
            sigma_j = np.mean(manhat_dist(cluster_j, centroid_j))
            scores_i.append((sigma_i + sigma_j) / manhat_dist(centroid_i, centroid_j))

        DB += 1 / k * np.max(scores_i)
    return DB


if __name__ == "__main__":

    X = np.array(
        [
            [5, 8],
            [6, 7],
            [6, 4],
            [5, 7],
            [5, 5],
            [6, 5],
            [1, 7],
            [7, 5],
            [6, 5],
            [6, 7],
        ]
    )
    centroids_init = uniform_init_centroids(3, X)
    # centroids_init = np.array([[7, 5], [9, 7], [9, 1]])

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids_init[:, 0], centroids_init[:, 1])
    plt.show()

    centroids, assignments = train_k_means(centroids_init, X, max_iter=30)
    print(davis_bouldin_index(centroids, X, assignments))

    plt.figure()
    for x_subset_indices in assignments.values():
        plt.scatter(X[x_subset_indices, 0], X[x_subset_indices, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.show()
