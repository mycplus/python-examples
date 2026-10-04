"""Top-down merge sort in Python 3."""


def merge_sort(a: list[int]) -> list[int]:
    """Return a new sorted list. Stable: on a tie the left element goes first."""
    if len(a) < 2:
        return list(a)
    mid = len(a) // 2
    left, right = merge_sort(a[:mid]), merge_sort(a[mid:])
    out: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if right[j] < left[i]:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


if __name__ == "__main__":
    data = [29, 10, 14, 37, 13, 5, 41, 22]
    print("Sorted:", merge_sort(data))
