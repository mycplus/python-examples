"""The article's worked example: seven labelled points, one query, four values of k."""

import numpy as np

from knn import KNNClassifier

points = np.array([[2, 4], [0, 4], [5, 4], [0, 1], [3, 6], [2, 3], [5, 5]], float)
labels = np.array(["A", "A", "A", "A", "B", "B", "B"])
query = np.array([[3.0, 3.0]])

model = KNNClassifier(k=7).fit(points, labels)
dist, idx = model.kneighbors(query)
print("query (3, 3); training points sorted by distance:")
print("rank  point   class  squared  distance")
for rank, (i, d) in enumerate(zip(idx[0], dist[0], strict=True), start=1):
    x, y = points[i]
    sq = (x - 3) ** 2 + (y - 3) ** 2
    print(f"{rank:4}  ({x:.0f}, {y:.0f})  {labels[i]:>5}  {sq:7.0f}  {d:8.3f}")

print("\n k  votes A  votes B  prediction")
for k in (1, 3, 5, 7):
    near = labels[idx[0, :k]]
    a, b = int((near == "A").sum()), int((near == "B").sum())
    pred = KNNClassifier(k=k).fit(points, labels).predict(query)[0]
    print(f"{k:2}  {a:7}  {b:7}  {pred:>10}")
