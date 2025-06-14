import numpy as np
from sklearn.cluster import KMeans

def cluster_features(features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans.cluster_centers_
