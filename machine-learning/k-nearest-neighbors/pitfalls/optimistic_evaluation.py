"""Pitfall: two ways to report a KNN accuracy that the model will not reach.

1. Scoring on the training rows: with k = 1 every row is its own nearest
   neighbor, so training accuracy is 100% whatever the data.
2. Choosing k by its score on the test rows, then reporting that score.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_wine_split, standardize
from knn import KNNClassifier

X_train, X_test, y_train, y_test, _ = load_wine_split()
S_train, S_test = standardize(X_train, X_test)

for name, A, B in (("unscaled", X_train, X_train), ("standardized", S_train, S_train)):
    pred = KNNClassifier(k=1).fit(A, y_train).predict(B)
    print(
        f"k=1, {name:12} accuracy on its own training rows: {(pred == y_train).mean():.3f}"
    )

scores = {}
for k in range(1, 32, 2):
    pred = KNNClassifier(k=k).fit(S_train, y_train).predict(S_test)
    scores[k] = int((pred == y_test).sum())
best = max(scores, key=lambda k: scores[k])
print(
    f"\nstandardized, test-set score for every odd k from 1 to 31 (of {len(y_test)}):"
)
print(" ".join(f"{k}:{c}" for k, c in scores.items()))
print(f"best k chosen on the test rows: k={best}, {scores[best]}/{len(y_test)}")
print(
    f"k=27 chosen by cross-validation on the training rows: {scores[27]}/{len(y_test)}"
)
