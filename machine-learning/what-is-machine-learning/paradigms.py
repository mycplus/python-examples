"""The other two paradigms on the same data and the same loop.

Supervised learning was first_model.py. This script shows what the other two
paradigms look like when they are stripped to the same size: clustering without
labels, and an agent learning from rewards. Both are seeded and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 0
SIZE = 4
RESULTS = Path(__file__).resolve().parent / "results"


def unsupervised() -> dict[str, object]:
    """No labels during fitting: find groups, then compare with the truth."""
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    model = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X)
    sizes = np.bincount(model.labels_)
    print("unsupervised: k-means on the 150 iris rows with no labels shown")
    print(f"cluster sizes          {sizes.tolist()}")
    print(f"silhouette score       {silhouette_score(X, model.labels_):.4f}")
    print(f"adjusted Rand index    {adjusted_rand_score(iris.target, model.labels_):.4f}")
    for cluster in range(3):
        counts = np.bincount(iris.target[model.labels_ == cluster], minlength=3)
        names = iris.target_names
        parts = ", ".join(f"{names[i]} {counts[i]}" for i in range(3) if counts[i])
        print(f"cluster {cluster} holds {parts}")
    return {
        "cluster_sizes": sizes.tolist(),
        "silhouette": float(silhouette_score(X, model.labels_)),
        "adjusted_rand": float(adjusted_rand_score(iris.target, model.labels_)),
        "labels": [int(v) for v in model.labels_],
    }


def q_learning() -> dict[str, object]:
    """A 4x4 gridworld, a goal at the far corner and a wall of penalties."""
    goals = (3, 3)
    holes = {(1, 1), (2, 2)}
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rng = np.random.default_rng(SEED)
    q = np.zeros((SIZE, SIZE, len(actions)))
    alpha, gamma, epsilon, episodes = 0.5, 0.9, 0.2, 300

    def step(state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
        dr, dc = actions[action]
        nxt = (min(max(state[0] + dr, 0), SIZE - 1), min(max(state[1] + dc, 0), SIZE - 1))
        if nxt == goals:
            return nxt, 1.0, True
        if nxt in holes:
            return nxt, -1.0, True
        return nxt, -0.01, False

    for _ in range(episodes):
        state = (0, 0)
        for _ in range(50):
            if rng.random() < epsilon:
                action = int(rng.integers(len(actions)))
            else:
                action = int(np.argmax(q[state[0], state[1]]))
            nxt, reward, done = step(state, action)
            best_next = 0.0 if done else float(np.max(q[nxt[0], nxt[1]]))
            q[state[0], state[1], action] += alpha * (
                reward + gamma * best_next - q[state[0], state[1], action]
            )
            state = nxt
            if done:
                break

    path, state = [(0, 0)], (0, 0)
    for _ in range(SIZE * SIZE):
        action = int(np.argmax(q[state[0], state[1]]))
        state, _, done = step(state, action)
        path.append(state)
        if done:
            break
    print()
    print(f"reinforcement learning: tabular Q-learning, {episodes} episodes, "
          f"{SIZE}x{SIZE} grid")
    print(f"greedy path from (0, 0)  {' to '.join(str(p) for p in path)}")
    print(f"steps in that path       {len(path) - 1}")
    print(f"states visited            {len({(r, c) for r in range(SIZE) for c in range(SIZE)})}")
    print(f"largest Q value           {q.max():.4f}")
    return {
        "path": [[int(r), int(c)] for r, c in path],
        "steps": len(path) - 1,
        "episodes": episodes,
        "max_q": float(q.max()),
        "holes": sorted([int(r), int(c)] for r, c in holes),
        "goals": [int(goals[0]), int(goals[1])],
    }


def main() -> None:
    payload = {"unsupervised": unsupervised(), "q_learning": q_learning()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "paradigms.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
