"""The article's worked example: eight customers, one feature, two impurity measures.

Gini and entropy rank the seven possible splits differently, so trees grown
with each one start with a different question.
"""

import numpy as np

from tree import Criterion, DecisionTreeClassifier, impurity

months = np.arange(1.0, 9.0)  # months as a customer
left_service = np.array([0, 0, 0, 0, 1, 0, 0, 1])  # 1 = cancelled

parent = np.bincount(left_service, minlength=2).astype(float)
print(f"all 8 customers: {int(parent[1])} cancelled")
print(
    f"gini {impurity(parent, 'gini'):.4f}, entropy {impurity(parent, 'entropy'):.4f} bits"
)

print("\nsplit         left    right   gini gain  entropy gain")
for i in range(1, 8):
    threshold = (months[i - 1] + months[i]) / 2
    a = np.bincount(left_service[:i], minlength=2).astype(float)
    b = np.bincount(left_service[i:], minlength=2).astype(float)
    gains = []
    for criterion in ("gini", "entropy"):
        child = (i * impurity(a, criterion) + (8 - i) * impurity(b, criterion)) / 8
        gains.append(float(impurity(parent, criterion) - child))
    print(
        f"months <= {threshold:.1f}  {int(a[1])} of {i}  {int(b[1])} of {8 - i}"
        f"  {gains[0]:9.4f}  {gains[1]:12.4f}"
    )

X = months.reshape(-1, 1)
criteria: tuple[Criterion, ...] = ("gini", "entropy")
print()
for criterion in criteria:
    root = DecisionTreeClassifier(criterion, max_depth=1).fit(X, left_service)
    print(f"{criterion:7} tree's first split: months <= {root.threshold_[0]:.1f}")
