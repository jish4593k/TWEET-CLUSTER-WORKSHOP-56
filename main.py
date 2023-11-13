import sys
import json
import re
import copy
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


 

def kmeans_scikit(data, k):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    centroids = vectorizer.inverse_transform(kmeans.cluster_centers_)
    
    return labels, centroids

def jaccard_distance(a, b):
    intersection = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return round(1 - (float(intersection) / union), 4)

def calculate_jaccard_distances(terms_all, centroids):
    distances = []
    for centroid in centroids:
        distances.append([jaccard_distance(tweet, centroid) for tweet in terms_all])
    return np.array(distances)

def assign_clusters(jaccard_distances):
    return np.argmin(jaccard_distances, axis=0)

def sse(jaccard_distances):
    return np.sum(np.power(jaccard_distances, 2))

def main():
    with open(str(sys.argv[3]), 'r') as file:
        tweets = [json.loads(line)['text'] for line in file]

    processed_tweets = [preprocess_text(tweet) for tweet in tweets]
    
    k = int(sys.argv[1])
    centroids_file_path = str(sys.argv[2])
    output_file_path = str(sys.argv[4])

    # Loading initial centroids from the file
    with open(centroids_file_path, 'r') as centroids_file:
        centroids = [int(line.strip()) for line in centroids_file]

    # K-means clustering using scikit-learn
    labels, new_centroids = kmeans_scikit(processed_tweets, k)

    # Output clusters and SSE
    with open(output_file_path, 'w') as output_file:
        output_clusters(labels, k, centroids, output_file)
        output_file.write(f'SSE: {calculate_sse(new_centroids, centroids, processed_tweets, k)}\n')

def output_clusters(labels, k, centroids, output_file):
    clusters = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx + 1)

    for i, cluster in enumerate(clusters):
        output_file.write(f'Cluster {i + 1}: {cluster}\n')

def calculate_sse(new_centroids, centroids, processed_tweets, k):
    new_jaccard_distances = calculate_jaccard_distances(processed_tweets, new_centroids)
    old_jaccard_distances = calculate_jaccard_distances(processed_tweets, centroids)
    
    return sse(new_jaccard_distances - old_jaccard_distances)

if __name__ == "__main__":
    main()
