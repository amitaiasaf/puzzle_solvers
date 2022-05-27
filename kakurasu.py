from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from itertools import chain, combinations
import re
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple, Union, Callable
import requests


def powerset(iterable: Iterable) -> Iterable[FrozenSet]:
    return chain.from_iterable(map(set, combinations(iterable, i)) for i in range(len(iterable)))


class KakurasuError(Exception):
    pass


class CellType(Enum):
    UNDETERMINED = 0
    BLACK = 1
    WHITE = 2


@dataclass
class Cell:
    type: CellType = CellType.UNDETERMINED

    def __str__(self) -> str:
        if self.type == CellType.UNDETERMINED:
            return ","
        elif self.type == CellType.BLACK:
            return "#"
        elif self.type == CellType.WHITE:
            return "-"


class Line:
    def __init__(self, length: int, black_sum: int) -> None:
        self.black_sum = black_sum
        self.remaining = set(range(1, length + 1))
        self.options = list(filter(self.filter_option_by_sum, powerset(self.remaining)))

    def filter_option_by_sum(self, option: FrozenSet[int]):
        return sum(option) == self.black_sum

    @staticmethod
    def in_option(index: int) -> Callable[[Set[int]], bool]:
        return lambda option: index in option

    @staticmethod
    def not_in_option(index: int) -> Callable[[Set[int]], bool]:
        return lambda option: index not in option

    def notify_value(self, index: int, value: CellType):
        if value == CellType.BLACK:
            self.options = list(filter(self.in_option(index + 1), self.options))
        elif value == CellType.WHITE:
            self.options = list(filter(self.not_in_option(index + 1), self.options))
        else:
            raise KakurasuError()
        self.remaining.discard(index + 1)

    def get_new_certain_values(self) -> Dict[int, CellType]:
        certain_values = {}
        for i in self.remaining:
            if all(map(self.in_option(i), self.options)):
                certain_values[i - 1] = CellType.BLACK
            elif all(map(self.not_in_option(i), self.options)):
                certain_values[i - 1] = CellType.WHITE
        return certain_values


class Kakurasu:
    def __init__(self, cells: Union(List[List[Cell]], None), cols_sums: List[int], rows_sums: List[int]):
        if cells is not None:
            self.cells: List[List[Cell]] = cells
            self.undetermined_count = sum(line.count(Cell()) for line in self.cells)
        else:
            self.cells = [[Cell() for _ in rows_sums] for _ in cols_sums]
            self.undetermined_count = len(cols_sums) * len(rows_sums)
        self.rows = [Line(len(cols_sums), black_sum) for black_sum in rows_sums]
        self.cols = [Line(len(rows_sums), black_sum) for black_sum in cols_sums]

    def __str__(self) -> str:
        lines = [[""] + list(map(str, range(1, len(self.cols) + 1))) + [""]]
        for i, row in enumerate(self.rows):
            lines.append([i + 1] + self.cells[i] + [row.black_sum])
        lines.append([""] + [col.black_sum for col in self.cols])
        return "\n".join("\t".join(map(str, line)) for line in lines)

    @staticmethod
    def from_internet(task: str, width: int, height: int) -> Kakurasu:
        sums = [int(s) for s in task.split("/")]
        assert len(sums) == width + height, "Invalid width or height"
        return Kakurasu(None, sums[:width], sums[width:])

    def __setitem__(self, key: Tuple[int, int], value: CellType):
        row, col = key
        if self.cells[row][col].type == value:
            return
        if self.cells[row][col].type != CellType.UNDETERMINED:
            raise KakurasuError()
        self.cells[row][col].type = value
        self.cols[col].notify_value(row, value)
        self.rows[row].notify_value(col, value)
        self.undetermined_count -= 1

    def solve(self):
        last_undetermined = 0
        while self.undetermined_count:
            if last_undetermined == self.undetermined_count:
                raise KakurasuError()

            for i, row in enumerate(self.rows):
                for j, value in row.get_new_certain_values().items():
                    self[i, j] = value

            for j, col in enumerate(self.cols):
                for i, value in col.get_new_certain_values().items():
                    self[i, j] = value


def load_puzzle_from_internet():
    request = requests.post("https://www.puzzle-kakurasu.com/?size=14")
    text = request.text

    task = re.search(r"var task = '([^']*)'", text).group(1)
    width = int(re.search(r'name="w" value="(\d+)"', text).group(1))
    height = int(re.search(r'name="h" value="(\d+)"', text).group(1))
    return task, width, height


puzzle = load_puzzle_from_internet()
kakurasu = Kakurasu.from_internet(*puzzle)
print(kakurasu)
kakurasu.solve()
print("Solved: ")
print(kakurasu)
