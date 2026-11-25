"""How much a single decision tree depends on details of its training data.

1. scikit-learn's DecisionTreeClassifier with 20 different random_state
   values, everything else at its defaults.
2. This article's tree on 200 bootstrap samples of the training rows, and
   the majority vote of those 200 trees (bagging, the idea behind random
   forests).
"""

from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnTree

from data import load_split
from tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test, names = load_split()

trees, scores = set(), []
for seed in range(20):
    model = SklearnTree(random_state=seed).fit(X_train, y_train)
    t = model.tree_
    trees.add((tuple(t.feature), tuple(np.round(t.threshold, 6))))
    scores.append(int((model.predict(X_test) == y_test).sum()))
print("scikit-learn, default settings, random_state = 0 to 19:")
print(
    f"  {len(trees)} different trees; test rows correct: {min(scores)} to {max(scores)} of 114"
)

rng = np.random.default_rng(2026)
roots: Counter[str] = Counter()
votes = np.zeros(len(y_test))
correct = []
for _ in range(200):
    rows = rng.integers(0, len(y_train), len(y_train))  # sample with replacement
    tree = DecisionTreeClassifier("gini").fit(X_train[rows], y_train[rows])
    roots[names[tree.feature_[0]]] += 1
    pred = tree.predict(X_test)
    votes += np.asarray(pred, dtype=float)
    correct.append(int((pred == y_test).sum()))
print("\nthis article's tree (gini, no depth limit) on 200 bootstrap samples:")
print("  feature tested at the root:")
for name, count in roots.most_common():
    print(f"    {name:22} {count:3}")
print(
    f"  test rows correct per tree: mean {np.mean(correct):.1f}, "
    f"min {min(correct)}, max {max(correct)} of 114"
)
bagged = (votes > 100).astype(int)  # majority of the 200 trees
print(f"  majority vote of the 200 trees: {int((bagged == y_test).sum())} of 114")
