"""Heap sort in Python 3."""


def sift_down(a: list[int], root: int, n: int) -> None:
    """Move a[root] down the max-heap a[0:n] until neither child is larger."""
    while True:
        child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and a[child] < a[child + 1]:
            child += 1
        if not a[root] < a[child]:
            return
        a[root], a[child] = a[child], a[root]
        root = child


def heap_sort(a: list[int]) -> list[int]:
    """Sort list a in place. O(n log n) worst case, not stable."""
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)
    return a


if __name__ == "__main__":
    data = [29, 10, 14, 37, 13, 5, 41, 22]
    print("Sorted:", heap_sort(data))
