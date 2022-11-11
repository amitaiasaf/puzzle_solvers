from __future__ import annotations
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import re
from typing import Callable

from utils.website_interface import WebsiteInterface


class StarBattleException(Exception):
    pass


class CellType(Enum):
    UNDETERMINED = 0
    STAR = 1
    NOT_STAR = 2


@dataclass
class Cell:
    type: CellType = CellType.UNDETERMINED

    def __str__(self) -> str:
        if self.type == CellType.UNDETERMINED:
            return " "
        elif self.type == CellType.STAR:
            return "#"
        elif self.type == CellType.NOT_STAR:
            return "*"


PuzzleDataType = list[list[Cell]]

point_to_neighbors = {}


@dataclass(frozen=True)
class Point:
    row: int
    col: int

    def get_neighbors(self) -> set[Point]:
        if self in point_to_neighbors:
            return point_to_neighbors[self]
        points = set()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    points.add(Point(self.row + i, self.col + j))
        point_to_neighbors[self] = points
        return points

    def is_neighbor(self, other: Point) -> bool:
        return abs(self.row - other.row) <= 1 and abs(self.col - other.col) <= 1


Option = frozenset[Point]

points_to_options = {}


class Block:
    def __init__(self,
                 points: set[Point],
                 stars: int,
                 options: list[Option] | None = None,
                 options_neighbors: dict[Option, set[Point]] | None = None,
                 line: bool = False) -> None:
        self.points = points
        self.stars = stars
        if options:
            self.options = options
        else:
            if not line:
                self.options: list[Option] = self.create_options(copy(self.points), self.stars)
            else:
                self.options: list[Option] = self.create_options_for_line(
                    sorted(self.points, key=lambda point: (point.row, point.col)), self.stars)

        self.options_neighbors = options_neighbors or {}

    @ classmethod
    def create_options_for_line(cls, points: list[Point], size) -> list[Option]:
        if size == 1:
            return [frozenset({point}) for point in points]
        if size * 2 - 1 > len(points):
            return []
        options: list[Option] = []
        for i, point in enumerate(points[:-(size * 2 - 2)]):
            cur_options = [frozenset({point} | option)
                           for option in cls.create_options_for_line(points[i + 2:], size - 1)]
            options += cur_options
        return options

    @ classmethod
    def create_options(cls, points: set[Point], size) -> list[Option]:
        if size == 1:
            return [frozenset({point}) for point in points]
        if size > len(points):
            return []
        options: list[Option] = []
        for point in sorted(points, key=lambda p: len(p.get_neighbors())):
            points.remove(point)
            possible_places = points - point.get_neighbors()
            cur_options = [frozenset({point} | option)
                           for option in cls.create_options(possible_places, size - 1)]
            options += cur_options
        return options

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __deepcopy__(self, _) -> Block:
        return Block(copy(self.points), self.stars, copy(self.options), self.options_neighbors)

    def get_option_neighbors(self, option: Option):
        option = frozenset(option)
        if option not in self.options_neighbors:
            self.options_neighbors[option] = set().union(*map(Point.get_neighbors, option))
        return self.options_neighbors[option]

    def is_valid_option(self, option: Option):
        for p, q in combinations(option, 2):
            if p.is_neighbor(q):
                return False
        return True

    @staticmethod
    def in_option(point: Point) -> Callable[[Option], bool]:
        return lambda option: point in option

    @staticmethod
    def not_in_option(point: Point) -> Callable[[Option], bool]:
        return lambda option: point not in option

    def notify_value(self, point: Point, value: CellType):
        if value == CellType.STAR:
            self.options = list(filter(self.in_option(point), self.options))
            self.stars -= 1
            if self.stars < 0:
                raise StarBattleException()
        elif value == CellType.NOT_STAR and point in self.points:
            self.options = list(filter(self.not_in_option(point), self.options))
        else:
            raise StarBattleException()
        self.points.discard(point)

    def get_star_points(self) -> set[Point]:
        return self.points.intersection(*self.options)

    def get_not_star_points(self) -> set[Point]:
        if not self.options:
            raise StarBattleException()
        if self.stars > len(self.points):
            raise StarBattleException()
        possible_stars = set().union(*self.options) & self.points
        not_stars = self.points - possible_stars
        neighbors_for_all_options = set().union(*(p.get_neighbors() for p in possible_stars))

        for option in self.options:
            if not neighbors_for_all_options:
                break
            neighbors_for_all_options.intersection_update(self.get_option_neighbors(option))

        return not_stars | neighbors_for_all_options


class StarBattle:
    def __init__(self,
                 size: int,
                 stars: int,
                 blocks: list[Block],
                 cells: PuzzleDataType = None,
                 rows: list[Block] = None,
                 cols: list[Block] = None,
                 undetermined_count: int = 0) -> None:

        self.size = size
        self.stars = stars
        self.blocks = blocks
        self.cells = cells if cells is not None else [[Cell() for i in range(size)] for j in range(size)]
        self.cells_to_blocks = self.create_cells_to_blocks_mapping()
        self.rows = rows or [Block(set(Point(i, j) for j in range(self.size)), self.stars, line=True)
                             for i in range(self.size)]
        self.cols = cols or [Block(set(Point(j, i) for j in range(self.size)), self.stars, line=True)
                             for i in range(self.size)]
        self.all_blocks = self.blocks + self.rows + self.cols
        self.undetermined_count = undetermined_count or self.size ** 2

    def __deepcopy__(self, _) -> StarBattle:
        return StarBattle(self.size,
                          self.stars,
                          deepcopy(self.blocks),
                          deepcopy(self.cells),
                          deepcopy(self.rows),
                          deepcopy(self.cols),
                          self.undetermined_count)

    def __str__(self) -> str:
        lines = []
        lines.append(" - ".join(" " * (self.size + 1)))
        for i, line in enumerate(self.cells):
            next_line_data = [" "]
            cur_line_data = ["|"]
            for j, cell in enumerate(line):
                cur_line_data.append(str(cell))
                if j == self.size - 1 or self.cells_to_blocks[i][j] is not self.cells_to_blocks[i][j + 1]:
                    cur_line_data.append("|")
                else:
                    cur_line_data.append(" ")
                if i == self.size - 1 or self.cells_to_blocks[i][j] is not self.cells_to_blocks[i + 1][j]:
                    next_line_data.append("-")
                else:
                    next_line_data.append(" ")
                next_line_data.append(" ")
            lines.append(" ".join(cur_line_data))
            lines.append(" ".join(next_line_data))
        return "\n".join(lines)

    def __getitem__(self, point: Point) -> Cell:
        return self.cells[point.row][point.col]

    @staticmethod
    def from_website(website_interface: StarBattleWebsiteInterface):
        assert website_interface.width == website_interface.height
        size = website_interface.width
        task = website_interface.task
        block_ids = list(map(int, task.split(",")))
        block_points: list[set[Point]] = [set() for _ in range(size)]
        for i, block_id in enumerate(block_ids):
            row = i // size
            col = i % size
            block_points[block_id - 1].add(Point(row, col))

        blocks = [Block(points, website_interface.stars) for points in block_points]
        return StarBattle(size, website_interface.stars, blocks)

    def serialize_solution(self) -> str:
        return "".join("".join("y" if c.type == CellType.STAR else "n" for c in row) for row in self.cells)

    def create_cells_to_blocks_mapping(self) -> list[list[None | Block]]:
        cells_to_blocks: list[list[None | Block]] = [[None] * self.size for i in range(self.size)]
        for block in self.blocks:
            for point in block.points:
                cells_to_blocks[point.row][point.col] = block

        return cells_to_blocks

    def mark_not_star(self, point: Point) -> None:
        if point.row < 0 or point.row >= self.size or point.col < 0 or point.col >= self.size:
            return
        if self[point].type == CellType.NOT_STAR:
            return
        if self[point].type == CellType.STAR:
            raise StarBattleException()
        self.cells[point.row][point.col].type = CellType.NOT_STAR
        self.undetermined_count -= 1
        self.cells_to_blocks[point.row][point.col].notify_value(point, CellType.NOT_STAR)
        self.rows[point.row].notify_value(point, CellType.NOT_STAR)
        self.cols[point.col].notify_value(point, CellType.NOT_STAR)

    def mark_star(self, point: Point) -> None:
        if point.row < 0 or point.row >= self.size or point.col < 0 or point.col >= self.size:
            raise StarBattleException()
        if self[point].type == CellType.STAR:
            return
        if self[point].type == CellType.NOT_STAR:
            raise StarBattleException()
        self.cells[point.row][point.col].type = CellType.STAR
        self.undetermined_count -= 1
        self.cells_to_blocks[point.row][point.col].notify_value(point, CellType.STAR)
        self.rows[point.row].notify_value(point, CellType.STAR)
        self.cols[point.col].notify_value(point, CellType.STAR)
        for neighbor in point.get_neighbors():
            self.mark_not_star(neighbor)

    def get_blocks_for_points(self, points: set[Point]) -> set[Block]:
        return set(self.cells_to_blocks[p.row][p.col] for p in points)

    def get_lines_points(self, lines: list[Block]) -> set[Point]:
        return set().union(*(line.points for line in lines))

    def find_best_block_to_guess(self):
        return min(self.all_blocks, key=lambda b: (len(b.options), -b.stars) if len(b.options) > 1 else (10000, 10000))

    def guess(self):
        guess_block = self.find_best_block_to_guess()
        guess = guess_block.options[-1]
        try:
            new_board = deepcopy(self)
            for p in guess:
                new_board.mark_star(p)
            new_board.solve()
            self.cells = new_board.cells
            self.undetermined_count = new_board.undetermined_count
        except StarBattleException:
            guess_block.options.remove(guess)

    def check_blocks_in_lines(self, lines: list[Block]) -> None:
        for i in range(len(lines)):
            row_points: set[Point] = set()
            row_blocks: set[Block] = set()
            rows_stars = 0
            for j in range(len(lines) - i):
                row_points |= lines[i + j].points
                row_blocks |= self.get_blocks_for_points(lines[i + j].points)
                rows_stars += lines[i + j].stars
                blocks_stars = sum(block.stars for block in row_blocks)
                if rows_stars == blocks_stars:
                    for point in set().union(*(block.points for block in row_blocks)).symmetric_difference(row_points):
                        self.mark_not_star(point)

    def solve(self) -> None:
        last_undetermined = self.undetermined_count
        while self.undetermined_count:
            for block in self.all_blocks[:]:
                if len(block.options) == 1:
                    self.all_blocks.remove(block)
                for point in block.get_star_points():
                    self.mark_star(point)
                for point in block.get_not_star_points():
                    self.mark_not_star(point)

            self.check_blocks_in_lines(self.rows)
            self.check_blocks_in_lines(self.cols)

            if last_undetermined == self.undetermined_count:
                self.guess()
            last_undetermined = self.undetermined_count


class StarBattleWebsiteInterface(WebsiteInterface):
    def __init__(self, url: str, params: dict = None) -> None:
        super().__init__(url, params)
        self.stars = int(re.search(r',stars:(\d+) ,', self.initial_response_text).group(1))


def main():
    website_interface = StarBattleWebsiteInterface(
        "https://www.puzzle-star-battle.com/", {"size": 9, "specific": 0, "specid": 123456})
    board = StarBattle.from_website(website_interface)
    board.solve()
    website_interface.submit_solution(board.serialize_solution())


if __name__ == "__main__":
    main()
