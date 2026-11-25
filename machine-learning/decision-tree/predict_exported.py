"""Load models/tree.txt and data/breast_cancer_test.csv with plain Python (no
NumPy, no scikit-learn) and print the report the C++ program must match.

The C++ companion (cpp-examples, machine-learning/decision-tree) reads the same
two files and is checked against the same tests/expected/tree_predict.txt.
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
Node = tuple[int, float, int, int, int]  # feature, threshold, left, right, class


def load_tree(path: Path) -> list[Node]:
    nodes: list[Node] = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        i, f, t, a, b, c = line.split()
        if int(i) != len(nodes):
            raise ValueError(f"{path}: node {i} out of order")
        nodes.append((int(f), float(t), int(a), int(b), int(c)))
    return nodes


def predict(nodes: list[Node], row: list[float]) -> tuple[int, list[int]]:
    """The predicted class and the nodes visited from the root to the leaf."""
    node, path = 0, [0]
    while nodes[node][0] >= 0:
        f, t, a, b, _ = nodes[node]
        node = a if row[f] <= t else b
        path.append(node)
    return nodes[node][4], path


def main() -> None:
    tree_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "models" / "tree.txt"
    csv_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else HERE / "data" / "breast_cancer_test.csv"
    )
    nodes = load_tree(tree_path)
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        names = next(reader)[1:]
        rows = [(int(r[0]), [float(v) for v in r[1:]]) for r in reader]

    leaves = sum(1 for n in nodes if n[0] < 0)
    print(f"tree: {len(nodes)} nodes, {leaves} leaves; test rows: {len(rows)}")
    preds = [predict(nodes, x)[0] for _, x in rows]
    cm = [[0, 0], [0, 0]]
    for (label, _), p in zip(rows, preds, strict=True):
        cm[label][p] += 1
    print(f"correct: {cm[0][0] + cm[1][1]} of {len(rows)}")
    print(f"benign:    {cm[0][0]:3} right, {cm[0][1]:2} called malignant")
    print(f"malignant: {cm[1][1]:3} right, {cm[1][0]:2} missed")
    print("predictions (1 = malignant):")
    text = "".join(str(p) for p in preds)
    for start in range(0, len(text), 57):
        print(f"  {text[start : start + 57]}")

    label, x = rows[0]
    pred, path = predict(nodes, x)
    print(f"test row 0 (class {label}) follows nodes {' '.join(map(str, path))}:")
    for node in path[:-1]:
        feature, t, _, _, _ = nodes[node]
        went = "<=" if x[feature] <= t else ">"
        print(f"  {names[feature]} = {x[feature]:.6g} {went} {t:.6g}")
    print(f"  leaf {path[-1]}: class {pred}")


if __name__ == "__main__":
    main()
