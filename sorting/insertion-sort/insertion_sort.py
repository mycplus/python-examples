"""Insertion sort in Python 3."""


def insertion_sort(a: list[int]) -> list[int]:
    """Sort list a in place. Stable: equal elements keep their order."""
    for i in range(1, len(a)):
        v = a[i]
        j = i
        while j > 0 and v < a[j - 1]:
            a[j] = a[j - 1]
            j -= 1
        a[j] = v
    return a


if __name__ == "__main__":
    data = [29, 10, 14, 37, 13, 5, 41, 22]
    print("Sorted:", insertion_sort(data))
