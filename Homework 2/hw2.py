from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 
import numpy as np
from scipy.stats import mode

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

visualizer = KElbowVisualizer(KMeans(n_init = "auto"), k=(2, 11))
visualizer.fit(X)

# TODO: Determine the best K for K-means
bestK = visualizer.elbow_value_

# TODO: Calculate accuracy for the best k
tsne = TSNE(n_components=2, init='random', random_state=0)
proj = tsne.fit_transform(X)
bestKMeans = KMeans(n_clusters=bestK, random_state=0)
clusters = bestKMeans.fit_predict(proj)

print(clusters.shape)
accuracy = accuracy_score(y_true, clusters)

# TODO: Draw a confusion matrix
matrix = confusion_matrix(y_true, clusters)

print(f"Best K: {bestK}")
print(f"Best K Accuracy: {accuracy}")
print("Confusion Matrix:")
print(matrix)
visualizer.show()

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_true, clusters)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0, 1 , 2, 3],
            yticklabels=[0, 1 , 2, 3])
plt.xlabel('true label')
plt.ylabel('predicted label');
