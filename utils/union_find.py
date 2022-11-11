from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, TypeVar


T = TypeVar('T', bound=Hashable)


class UnionFindLeaf(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
        self.size = 1
        self.parent = self


class UnionFind(Generic[T]):
    def __init__(self, values: Iterable[T]) -> None:
        self.values = {value: UnionFindLeaf(value) for value in values}

    def __deepcopy__(self, _) -> UnionFind[T]:
        new_object = UnionFind(self.values)
        for value, leaf in new_object.values.items():
            leaf.size = self.values[value].size
            leaf.parent = new_object.values[self.values[value].parent.value]
        return new_object

    def find(self, value: T) -> UnionFindLeaf[T]:
        root = self.values[value]
        while root.parent != root:
            root, root.parent = root.parent, root.parent.parent
        return root

    def union(self, value1: T, value2: T) -> None:
        root1 = self.find(value1)
        root2 = self.find(value2)
        if root1 is root2:
            return
        if root1.size < root2.size:
            root1, root2 = root2, root1

        root2.parent = root1
        root1.size += root2.size

    def are_in_the_same_component(self, value1: T, value2: T) -> bool:
        return self.find(value1) is self.find(value2)
