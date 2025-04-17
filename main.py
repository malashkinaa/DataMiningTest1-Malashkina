import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin, silhouette_score

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
        # Призначення точок до найближчого центроїда
        labels = pairwise_distances_argmin(X, centroids)
        
        # Оновлення центроїдів
        new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else X[np.random.randint(0, X.shape[0])]
                                  for i in range(k)])
        
        # Перевірка на стабільність
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Elbow метод для вибору оптимального k
inertia_values = []
silhouette_values = []
k_values = range(1, min(11, X.shape[0]))  # Максимум k має бути менше або рівне кількості точок

for k in k_values:
    centroids, labels = kmeans(X, k)
    inertia = sum(np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k))
    inertia_values.append(inertia)
    
    if k > 1 and len(np.unique(labels)) > 1:  # Для розрахунку силуету потрібно більше 1 кластеру
        silhouette = silhouette_score(X, labels)
        silhouette_values.append(silhouette)
    else:
        silhouette_values.append(None)  # Для k = 1 або k, коли є лише 1 кластер, силует не розраховується

# Побудова Elbow-графу
plt.figure(figsize=(12, 6))

# Elbow-графік
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Сумарна внутрішньокластерна дисперсія (Inertia)')
plt.title('Elbow метод для визначення оптимального k')
plt.grid(True)

# Графік для силуету
plt.subplot(1, 2, 2)
plt.plot(k_values[1:], silhouette_values[1:], marker='o', color='orange')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Сила методу (Silhouette Score)')
plt.title('Silhouette Score для оптимального k')
plt.grid(True)

plt.tight_layout()
plt.show()
