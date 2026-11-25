"""Train the tree chosen in breast_cancer_experiment.py (entropy, depth 3) on
all 455 training rows and write the files the C++ companion uses:

  models/tree.txt                the node table, loaded at run time
  models/breast_cancer_tree.hpp  the same tree as nested ifs in C++
  data/breast_cancer_test.csv    the 114 test rows
"""

from pathlib import Path

from data import HERE, load_split, write_test_csv
from export import save_tree, to_cpp
from tree import DecisionTreeClassifier


def main(out: Path = HERE) -> None:
    X_train, _, y_train, _, names = load_split()
    tree = DecisionTreeClassifier("entropy", max_depth=3).fit(X_train, y_train)
    save_tree(tree, out / "models" / "tree.txt")
    (out / "models" / "breast_cancer_tree.hpp").write_text(
        to_cpp(tree, names, "breast_cancer_tree"), newline="\n"
    )
    write_test_csv(out / "data" / "breast_cancer_test.csv")


if __name__ == "__main__":
    main()
