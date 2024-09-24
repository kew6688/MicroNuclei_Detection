import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def boxToCenters(boxes):
  """
  A list of box in ([[x1,y1,x2,y2],...]) 
    to a list of centers [[x,y],...]
  """
  centers = (boxes[:,0:2] + boxes[:,2:])/2
  return centers.int().cpu().numpy()

def resolveApop(boxes, thresh=5, eps=20, min_samples=1):
    centers = boxToCenters(boxes)
    if len(centers) <= 0:
      return 0
      
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # number of mn without apop cluster
    cnt = 0
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1: continue
        n_in_cluster = sum(labels == k)
        if n_in_cluster <= thresh: 
            cnt += n_in_cluster
    return cnt

def testApop(boxes,eps=20, min_samples=1):
    X = boxToCenters(boxes)
    # X = StandardScaler().fit_transform(centers)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()