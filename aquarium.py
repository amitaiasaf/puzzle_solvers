from __future__ import annotations
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from itertools import chain, combinations
from typing import Callable, Iterable

from numpy import block

from website_interface import WebsiteInterface


class AquariumException(Exception):
    pass


def powerset(iterable: Iterable) -> Iterable[frozenset]:
    return chain.from_iterable(map(set, combinations(iterable, i)) for i in range(len(iterable)))


@dataclass(frozen=True)
class Point:
    row: int
    col: int


@dataclass
class Cell(Enum):
    UNDETERMINED = " "
    WATER = "#"
    AIR = ","

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        return self is other


PuzzleDataType = list[list[Cell]]


Points = set[Point]


@dataclass
class Block:
    points: Points

    def get_points_to_propagate(self, row: int, value: Cell):
        if value == Cell.AIR:
            return filter(lambda point: point.row <= row, self.points)
        elif value == Cell.WATER:
            return filter(lambda point: point.row >= row, self.points)
        raise AquariumException()


@dataclass
class Row:
    def __init__(self,
                 block_ids: list[int],
                 number: int,
                 block_to_indices: dict[int, list[int]] = None,
                 options: list[set[int]] = None) -> None:
        self.block_ids = block_ids
        self.number = number
        if block_to_indices is None:
            self.block_to_indices: dict[int, list[int]] = {}
            for i, block_id in enumerate(self.block_ids):
                if block_id not in self.block_to_indices:
                    self.block_to_indices[block_id] = []
                self.block_to_indices[block_id].append(i)
        else:
            self.block_to_indices = block_to_indices
        self.options = options or list(filter(self.filter_option_by_count, powerset(self.block_to_indices)))

    def __deepcopy__(self, _) -> Row:
        return Row(self.block_ids, self.number, copy(self.block_to_indices), copy(self.options))

    def filter_option_by_count(self, option: set[int]):
        return sum(len(self.block_to_indices[b]) for b in option) == self.number

    @staticmethod
    def in_option(index: int) -> Callable[[int], bool]:
        return lambda option: index in option

    @staticmethod
    def not_in_option(index: int) -> Callable[[int], bool]:
        return lambda option: index not in option

    def get_new_certain_values(self) -> dict[int, Cell]:
        if not self.options:
            raise AquariumException()
        certain_values = {}
        for block_id in self.block_to_indices:
            water_in_all_options = True
            air_in_all_options = True
            for option in self.options:
                water_in_all_options &= block_id in option
                air_in_all_options &= block_id not in option
                if not water_in_all_options and not air_in_all_options:
                    break
            if water_in_all_options:
                for i in self.block_to_indices[block_id]:
                    certain_values[i] = Cell.WATER
            elif air_in_all_options:
                for i in self.block_to_indices[block_id]:
                    certain_values[i] = Cell.AIR
        return certain_values

    def notify_value(self, index: int, value: Cell):
        block_id = self.block_ids[index]
        if block_id in self.block_to_indices:
            del self.block_to_indices[block_id]
            if value == Cell.WATER:
                self.options = list(filter(self.in_option(block_id), self.options))
            elif value == Cell.AIR:
                self.options = list(filter(self.not_in_option(block_id), self.options))
            else:
                raise AquariumException()


class Col:
    def __init__(self,
                 block_ids: list[int],
                 number: int,
                 remaining_cells: int = -1,
                 remaining_water: int = -1,
                 block_to_indices: dict[int, list[int]] = None) -> None:
        self.block_ids = block_ids
        self.number = number
        self.remaining_cells = remaining_cells if remaining_cells != -1 else len(block_ids)
        self.remaining_water = remaining_water if remaining_water != -1 else number
        if block_to_indices:
            self.block_to_indices = block_to_indices
        else:
            self.block_to_indices: dict[int, list[int]] = {}
            for i, block_id in enumerate(self.block_ids):
                if block_id not in self.block_to_indices:
                    self.block_to_indices[block_id] = []
                self.block_to_indices[block_id].append(i)

    def __deepcopy__(self, _) -> Col:
        return Col(self.block_ids,
                   self.number,
                   self.remaining_cells,
                   self.remaining_water,
                   deepcopy(self.block_to_indices))

    def notify_value(self, index: int, value: Cell):
        block_id = self.block_ids[index]

        self.block_to_indices[block_id].remove(index)
        self.remaining_cells -= 1
        if value == Cell.WATER:
            self.remaining_water -= 1
            if self.remaining_water < 0:
                raise AquariumException()

    def get_new_certain_values(self) -> dict[int, Cell]:
        certain_values = {}
        for block_id, indices in self.block_to_indices.items():
            if len(indices) >= self.remaining_water:
                for index in indices[:-self.remaining_water]:
                    certain_values[index] = Cell.AIR
            if len(indices) >= self.remaining_cells - self.remaining_water:
                for index in indices[self.remaining_cells-self.remaining_water:]:
                    certain_values[index] = Cell.WATER
        return certain_values


class Aquarium:
    def __init__(self,
                 size: int,
                 blocks: list[Block],
                 rows: list[Row],
                 cols: list[Col],
                 cells: PuzzleDataType,
                 cells_to_blocks: list[list[Block]] = None,
                 undetermined_count: int = 0) -> None:
        self.size = size
        self.blocks = blocks
        self.rows = rows
        self.cols = cols
        self.cells = cells or [[Cell() for i in range(size)] for j in range(size)]
        self.cells_to_blocks = cells_to_blocks or self.create_cells_to_blocks_mapping()
        self.undetermined_count = undetermined_count or size ** 2

    def __deepcopy__(self, _) -> Aquarium:
        return Aquarium(self.size,
                        self.blocks,
                        deepcopy(self.rows),
                        deepcopy(self.cols),
                        deepcopy(self.cells),
                        self.cells_to_blocks,
                        self.undetermined_count)

    def __str__(self) -> str:
        lines = []
        lines.append("   " + "".join(f"  {i:2d} " for i in range(1, self.size + 1)))
        lines.append("  " + "  --".join(" " * (self.size + 1)))
        for i, line in enumerate(self.cells):
            next_line_data = ["    "]
            cur_line_data = [f"{i+1:2d}", "| "]
            for j, cell in enumerate(line):
                cur_line_data.append(str(cell))
                if j == self.size - 1 or self.cells_to_blocks[i][j] is not self.cells_to_blocks[i][j + 1]:
                    cur_line_data.append("| ")
                else:
                    cur_line_data.append("  ")
                if i == self.size - 1 or self.cells_to_blocks[i][j] is not self.cells_to_blocks[i + 1][j]:
                    next_line_data.append("--")
                else:
                    next_line_data.append("  ")
                next_line_data.append(" ")
            cur_line_data.append(f"{self.rows[i].number: 2d}")
            lines.append(" ".join(cur_line_data))
            lines.append(" ".join(next_line_data))
        lines.append("   " + "".join(f"  {col.number:2d} " for col in self.cols))
        return "\n".join(lines)

    def __getitem__(self, key: Point) -> Cell:
        return self.cells[key.row][key.col]

    def mark_value(self, point: Point, value: Cell, propagate: bool = True):
        if self[point] == value:
            return
        if self[point] != Cell.UNDETERMINED:
            raise AquariumException()
        self.undetermined_count -= 1
        self.cells[point.row][point.col] = value
        self.rows[point.row].notify_value(point.col, value)
        self.cols[point.col].notify_value(point.row, value)
        if propagate:
            for p in self.cells_to_blocks[point.row][point.col].get_points_to_propagate(point.row, value):
                self.mark_value(p, value, False)

    def create_cells_to_blocks_mapping(self) -> list[list[Block]]:
        cells_to_blocks = [[None] * self.size for i in range(self.size)]
        for block in self.blocks:
            for point in block.points:
                cells_to_blocks[point.row][point.col] = block

        return cells_to_blocks

    @staticmethod
    def from_website(website_interface: WebsiteInterface) -> Aquarium:
        assert website_interface.width == website_interface.height
        size = website_interface.width
        task = website_interface.task
        side, blocks_data = task.split(";")
        side_numbers = list(map(int, side.split("_")))
        block_ids = list(map(int, blocks_data.split(",")))
        block_points: dict[int, set[Point]] = defaultdict(set)
        for i, block_id in enumerate(block_ids):
            row = i // size
            col = i % size
            block_points[block_id - 1].add(Point(row, col))

        blocks = [Block(points) for points in block_points.values()]
        cells = [[Cell.UNDETERMINED for i in range(size)] for j in range(size)]
        rows = [Row(block_ids[i * size: (i + 1) * size], side_numbers[i + size]) for i in range(size)]
        cols = [Col(block_ids[i::size], side_numbers[i]) for i in range(size)]
        return Aquarium(size, blocks, rows, cols, cells)

    def serialize_solution(self) -> str:
        return "".join("".join("y" if c == Cell.WATER else "n" for c in row) for row in self.cells)

    def find_best_row_to_guess(self) -> int:
        return min(range(self.size), key=lambda i: len(self.rows[i].options) if len(self.rows[i].options) > 1 else 10000)

    def guess(self):
        guess_row = self.find_best_row_to_guess()
        guess = self.rows[guess_row].options[-1]
        try:
            new_board = deepcopy(self)
            new_board.rows[guess_row].options = [guess]
            new_board.solve()
            self.cells = new_board.cells
            self.undetermined_count = new_board.undetermined_count
        except AquariumException:
            self.rows[guess_row].options.pop()

    def solve(self):
        while self.undetermined_count:
            last_undetermined = self.undetermined_count

            for i, row in enumerate(self.rows):
                for j, value in row.get_new_certain_values().items():
                    self.mark_value(Point(i, j), value)

            for j, col in enumerate(self.cols):
                for i, value in col.get_new_certain_values().items():
                    self.mark_value(Point(i, j), value)

            if last_undetermined == self.undetermined_count:
                self.guess()


def main():
    website_interface = WebsiteInterface("https://www.puzzle-aquarium.com/", {"size": 9, "specific": 0, "specid": 2})
    board = Aquarium.from_website(website_interface)
    board.solve()
    website_interface.submit_solution(board.serialize_solution())


if __name__ == "__main__":
    main()
