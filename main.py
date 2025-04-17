import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

# Зчитуємо дані
data = pd.read_csv("synthetic_customers.csv")
X = data.values

# Функція для ініціалізації центроїдів (k-means++)
def initialize_centroids_kmeanspp(X, k):
    n_samples = X.shape[0]
    centroids = []
    
    # Випадковий перший центроїд
    idx = np.random.randint(0, n_samples)
    centroids.append(X[idx])
    
    for _ in range(1, k):
        dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        probabilities = dist_sq / np.sum(dist_sq)
        cumulative_prob = np.cumsum(probabilities)
        r = np.random.rand()
        for j, p in enumerate(cumulative_prob):
            if r < p:
                centroids.append(X[j])
                break
    return np.array(centroids)

# Основний K-Means алгоритм
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids_kmeanspp(X, k)
    
    for _ in range(max_iters):
        labels = pairwise_distances_argmin(X, centroids)
        new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
                                  for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Elbow метод
inertia_values = []
k_values = range(1, 10)

for k in k_values:
    centroids, labels = kmeans(X, k)
    inertia = sum(np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k))
    inertia_values.append(inertia)

# Побудова Elbow-графу
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Сумарна внутрішньокластерна дисперсія (Inertia)')
plt.title('Elbow метод для визначення оптимального k')
plt.grid(True)
plt.show()
