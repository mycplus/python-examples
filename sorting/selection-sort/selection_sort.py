"""Selection sort in Python 3."""


def selection_sort(a: list[int]) -> list[int]:
    """Sort list a in place with at most len(a) - 1 swaps. Not stable."""
    n = len(a)
    for i in range(n - 1):
        m = min(range(i, n), key=a.__getitem__)   # index of the first minimum
        if m != i:
            a[i], a[m] = a[m], a[i]
    return a


if __name__ == "__main__":
    data = [29, 10, 14, 37, 13, 5, 41, 22]
    print("Sorted:", selection_sort(data))
