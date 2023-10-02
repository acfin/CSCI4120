from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2, 10))
visualizer.fit(X)

# TODO: Determine the best K for K-means
bestK = visualizer.elbow_value_
bestKMeans = KMeans(n_clusters = bestK, random_state=0)
bestKMeans.fit(X)

# TODO: Calculate accuracy for the best K
y_pred = bestKMeans.labels_
accuracy = accuracy_score(y_true, y_pred)

# TODO: Draw a confusion matrix
matrix = confusion_matrix(y_true, y_pred)

print(f"Best K: {bestK}")
print(f"Best K Accuracy: {accuracy}")
print("Confusion Matrix:")
print(matrix)
visualizer.show()