from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from utils.website_interface import WebsiteInterface


class TentsException(Exception):
    pass


class CellType(Enum):
    UNDETERMINED = ","
    GRASS = "-"
    TREE = "T"
    TENT = "A"


@dataclass
class Cell:
    type: CellType = CellType.UNDETERMINED

    def __str__(self) -> str:
        return self.type.value


@dataclass(frozen=True)
class Point:
    row: int
    col: int

    def get_neighbors(self) -> set[Point]:
        points = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    points.append(Point(self.row + i, self.col + j))
        return points

    def get_non_diagonal_neighbors(self):
        return [Point(self.row - 1, self.col), Point(self.row + 1, self.col),
                Point(self.row, self.col - 1), Point(self.row, self.col + 1)]

    def is_neighbor(self, other: Point) -> bool:
        return abs(self.row - other.row) <= 1 and abs(self.col - other.col) <= 1


class Tree:
    def __init__(self, point: Point, size: int) -> None:
        self.point = point
        self.options = set()
        for option in point.get_non_diagonal_neighbors():
            if 0 <= option.row < size and 0 <= option.col < size:
                self.options.add(option)

        self.tent = None

    def __hash__(self) -> int:
        return hash(self.point)

    def remove_options(self, board: Tents) -> None:
        if self.tent:
            return
        for option in list(self.options):
            if option in board.tents_to_trees:
                self.options.remove(option)
            elif board[option].type not in [CellType.UNDETERMINED, CellType.TENT]:
                self.options.remove(option)
        if len(self.options) == 1:
            tent = self.options.pop()
            self.set_tent(tent, board)
            board.mark_tent(self.tent)

    def set_tent(self, tent: Point, board: Tents):
        if self.tent:
            raise TentsException()
        self.tent = tent
        board.tents_to_trees[self.tent] = self
        del board.trees[self.point]


class Line:
    neighbors_for_option: dict[frozenset[int], set[int]] = {}

    def __init__(self, size: int, tents_count: int, index: int, is_col: bool) -> None:
        self.size = size
        self.index = index
        self.is_col = is_col
        self.tents_count = tents_count
        self.options: list[set[int]] = None
        self.possible_tents = set(range(size))

    @staticmethod
    def in_option(index: int) -> Callable[[set[int]], bool]:
        return lambda option: index in option

    @staticmethod
    def not_in_option(index: int) -> Callable[[set[int]], bool]:
        return lambda option: index not in option

    def notify_not_tent(self, index: int):
        if index in self.possible_tents:
            self.possible_tents.remove(index)
            if self.options is not None:
                self.options = list(filter(self.not_in_option(index), self.options))
                if not self.options:
                    raise TentsException()

    def notify_tent(self, index: int):
        if index not in self.possible_tents:
            raise TentsException()
        if self.options is not None:
            self.options = list(filter(self.in_option(index), self.options))
        self.tents_count -= 1
        if self.tents_count < 0:
            raise TentsException()

    @classmethod
    def get_neighbors_for_option(cls, option) -> set[int]:
        option = frozenset(option)
        if option not in cls.neighbors_for_option:
            cls.neighbors_for_option[option] = set().union(*({i - 1, i, i + 1} for i in option))
        return cls.neighbors_for_option[option]

    def get_neighbors_for_all_options(self) -> set[int]:
        return set(range(self.size)).intersection(*(self.get_neighbors_for_option(option) for option in self.options))

    @classmethod
    def create_options_for_line(cls, indices: list[int], size: int) -> list[set[int]]:
        if size == 0:
            return [set()]
        if size > len(indices):
            return []
        options: list[set[int]] = []
        for i, index in enumerate(indices):
            if i < len(indices) - 1 and indices[i + 1] - index == 1:
                i += 1
            cur_options = [{index} | option for option in cls.create_options_for_line(indices[i + 1:], size - 1)]
            options += cur_options
        return options

    def init_options(self):
        if not self.options:
            self.options = self.create_options_for_line(sorted(self.possible_tents), self.tents_count)

    def get_grass_indices(self):
        return self.possible_tents - set().union(*self.options)

    def get_tent_indices(self):
        return self.possible_tents.intersection(*self.options)


class Tents:
    def __init__(self,
                 size: int,
                 cells: list[list[Cell]],
                 rows: list[Line],
                 cols: list[Line],
                 trees: dict[Point, Tree] = None,
                 undetermined_places: set[Point] = None,
                 tents_to_trees: dict[Point, Tree] = None) -> None:
        self.size = size
        self.cells = cells
        self.rows = rows
        self.cols = cols
        if not trees:
            self.trees: dict[Point, Tree] = {}
            self.undetermined_places: set[Point] = set()
            self.tents_to_trees: dict[Point, Tree] = {}
            for i in range(size):
                for j in range(size):
                    p = Point(i, j)
                    if self[p].type == CellType.TREE:
                        self.trees[p] = Tree(p, size)
                        self.rows[i].notify_not_tent(j)
                        self.cols[j].notify_not_tent(i)
                    else:
                        self.undetermined_places.add(p)
        else:
            self.trees = trees
            self.undetermined_places = undetermined_places
            self.tents_to_trees = tents_to_trees

    def __deepcopy__(self, _) -> Tents:
        return Tents(self.size,
                     deepcopy(self.cells),
                     deepcopy(self.rows),
                     deepcopy(self.cols),
                     deepcopy(self.trees),
                     deepcopy(self.undetermined_places),
                     deepcopy(self.tents_to_trees))

    def __str__(self) -> str:
        lines = [[""] + list(map(str, range(1, len(self.cols) + 1))) + [""]]
        for i, row in enumerate(self.rows):
            lines.append([i + 1] + self.cells[i] + [row.tents_count])
        lines.append([""] + [col.tents_count for col in self.cols])
        return "\n".join("\t".join(map(str, line)) for line in lines)

    def serialize_solution(self) -> str:
        return "".join("".join("y" if c.type == CellType.TENT else "n" for c in row) for row in self.cells)

    def __getitem__(self, point: Point) -> Cell:
        return self.cells[point.row][point.col]

    @staticmethod
    def from_website(website_interface: WebsiteInterface):
        assert website_interface.width == website_interface.height
        size = website_interface.width
        task = website_interface.task
        cells_part, board_numbers = task.split(",", 1)
        cells = [[Cell() for i in range(size)] for j in range(size)]
        cur_board_index = -1
        for c in cells_part:
            if c == "_":
                cur_board_index += 1
            elif "a" <= c <= "z":
                cur_board_index += ord(c) - ord("a") + 2
            if cur_board_index == size ** 2:
                break
            col = cur_board_index % size
            row = cur_board_index // size
            cells[row][col].type = CellType.TREE
        side_numbers = [int(n) if n else None for n in board_numbers.split(",")]
        cols = []
        rows = []
        for i in range(size):
            cols.append(Line(size, side_numbers[i], i, True))
            rows.append(Line(size, side_numbers[size + i], i, False))

        return Tents(size, cells, rows, cols)

    def get_places_without_trees(self) -> set[Point]:
        return self.undetermined_places - set().union(*(tree.options for tree in self.trees.values()))

    def mark_grass(self, point: Point) -> None:
        if point.row < 0 or point.row >= self.size or point.col < 0 or point.col >= self.size:
            return
        if self[point].type == CellType.TENT:
            raise TentsException()
        if self[point].type == CellType.UNDETERMINED:
            self.undetermined_places.remove(point)
            self[point].type = CellType.GRASS
            self.rows[point.row].notify_not_tent(point.col)
            self.cols[point.col].notify_not_tent(point.row)

    def mark_tent(self, point: Point) -> None:
        if self[point].type not in [CellType.TENT, CellType.UNDETERMINED]:
            raise TentsException()
        if self[point].type == CellType.UNDETERMINED:
            self.undetermined_places.remove(point)
            self[point].type = CellType.TENT
            self.rows[point.row].notify_tent(point.col)
            self.cols[point.col].notify_tent(point.row)
            if point not in self.tents_to_trees:
                possible_trees = set(self.trees.keys()).intersection(point.get_non_diagonal_neighbors())
                if not possible_trees:
                    raise TentsException()
                if len(possible_trees) == 1:
                    tree = self.trees[possible_trees.pop()]
                    tree.set_tent(point, self)

    def find_best_line_to_guess(self):
        return min(self.rows + self.cols, key=lambda line: len(line.options) if len(line.options) > 1 else 10000)

    def guess(self) -> None:
        guess_line = self.find_best_line_to_guess()
        guess = guess_line.options[-1]
        try:
            new_board = deepcopy(self)
            for j in guess:
                if guess_line.is_col:
                    new_board.mark_tent(Point(j, guess_line.index))
                else:
                    new_board.mark_tent(Point(guess_line.index, j))
            new_board.solve()
            self.cells = new_board.cells
            self.undetermined_places = new_board.undetermined_places
        except TentsException:
            guess_line.options.pop()

    def solve(self) -> None:
        for point in self.get_places_without_trees():
            self.mark_grass(point)
        for line in self.rows + self.cols:
            line.init_options()

        undetermined_count = len(self.undetermined_places)
        while self.undetermined_places:
            for i in range(self.size):
                for j in self.rows[i].get_grass_indices():
                    self.mark_grass(Point(i, j))
                for j in self.rows[i].get_tent_indices():
                    self.mark_tent(Point(i, j))
                for j in self.cols[i].get_grass_indices():
                    self.mark_grass(Point(j, i))
                for j in self.cols[i].get_tent_indices():
                    self.mark_tent(Point(j, i))

                for j in self.rows[i].get_neighbors_for_all_options():
                    if i > 0:
                        self.mark_grass(Point(i - 1, j))
                    if i + 1 < self.size:
                        self.mark_grass(Point(i + 1, j))

                for j in self.cols[i].get_neighbors_for_all_options():
                    if i > 0:
                        self.mark_grass(Point(j, i - 1))
                    if i + 1 < self.size:
                        self.mark_grass(Point(j, i + 1))

            for tree in list(self.trees.values()):
                tree.remove_options(self)

            for point in self.get_places_without_trees():
                self.mark_grass(point)

            if len(self.undetermined_places) == undetermined_count:
                self.guess()
            undetermined_count = len(self.undetermined_places)


def main():
    website_interface = WebsiteInterface("https://www.puzzle-tents.com/", {"size": 10, "specific": 0, "specid": 2})
    board = Tents.from_website(website_interface)
    board.solve()
    website_interface.submit_solution(board.serialize_solution())


if __name__ == "__main__":
    main()
