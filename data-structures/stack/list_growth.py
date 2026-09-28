"""Count how often a list's allocation changes over a million appends.

sys.getsizeof() reports the list object plus its pointer array, so a
change in the reported size marks a reallocation of that array.
"""
import sys

items: list[int] = []
last = sys.getsizeof(items)
changes = 0
for i in range(1_000_000):
    items.append(i)
    size = sys.getsizeof(items)
    if size != last:
        changes += 1
        last = size
print(f"list: {changes} reallocations for 1000000 appends "
      f"(CPython {sys.version_info.major}.{sys.version_info.minor})")
