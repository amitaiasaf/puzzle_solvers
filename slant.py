from __future__ import annotations
from copy import copy, deepcopy

from dataclasses import dataclass
from enum import Enum
from itertools import product
import random
from typing import Literal, Optional, Tuple
from utils.website_interface import WebsiteInterface
from utils.union_find import UnionFind


class SlantFatalException(Exception):
    pass


class SlantException(Exception):
    pass


def dict_symmetric_difference(d1: dict, d2: dict) -> dict:
    d = d1.copy()
    d.update(d2)
    for key in d1.keys() & d2.keys():
        del d[key]
    return d


class Diagonal(Enum):
    UNDETERMINED = " "
    SLASH = "/"
    BACKSLASH = "\\"

    def __str__(self) -> str:
        return self.value

    def __invert__(self) -> FilledDiagonal:
        if self == Diagonal.UNDETERMINED:
            raise SlantFatalException()
        elif self == Diagonal.SLASH:
            return Diagonal.BACKSLASH
        else:
            return Diagonal.SLASH

    def __deepcopy__(self, _) -> Diagonal:
        return self


FilledDiagonal = Literal[Diagonal.BACKSLASH, Diagonal.SLASH]


@dataclass(frozen=True)
class Point:
    row: int
    col: int

    def get_neighbors(self) -> list[Corner]:
        return [Corner(self.row, self.col), Corner(self.row, self.col + 1),
                Corner(self.row + 1, self.col), Corner(self.row + 1, self.col + 1)]

    def get_corners(self, diagonal: FilledDiagonal) -> tuple[Corner, Corner]:
        if diagonal == Diagonal.SLASH:
            return Corner(self.row + 1, self.col), Corner(self.row, self.col + 1)
        else:
            return Corner(self.row, self.col), Corner(self.row + 1, self.col + 1)

    def get_direction_for_corner(self, corner: Corner) -> FilledDiagonal:
        if (corner.row - self.row) == (corner.col - self.col) and abs(corner.row - self.row) <= 1:
            return Diagonal.BACKSLASH
        elif abs(corner.row - self.row) + abs(corner.col - self.col) == 1:
            return Diagonal.SLASH
        raise SlantFatalException()

    def __deepcopy__(self, _) -> Point:
        return self


@dataclass(frozen=True)
class Corner:
    row: int
    col: int

    def __deepcopy__(self, _) -> Corner:
        return self


class Number:
    def __init__(self,
                 corner: Corner,
                 number: int,
                 size: int,
                 diagonals: Optional[dict[Point, FilledDiagonal]] = None,
                 remaining: Optional[int] = None) -> None:
        self.corner = corner
        self.number = number
        self.size = size
        self.diagonals = diagonals if diagonals is not None else self.get_diagonal_neighbors()
        self.remaining = remaining if remaining is not None else number

    def __deepcopy__(self, _) -> Number:
        return Number(self.corner,
                      self.number,
                      self.size,
                      copy(self.diagonals),
                      self.remaining)

    def __str__(self) -> str:
        return str(self.number)

    def get_diagonal_neighbors(self) -> dict[Point, FilledDiagonal]:
        return self.get_upper_left_neighbor() | self.get_upper_right_neighbor() \
            | self.get_lower_left_neighbor() | self.get_lower_right_neighbor()

    def get_upper_neighbors(self) -> dict[Point, FilledDiagonal]:
        return self.get_upper_left_neighbor() | self.get_upper_right_neighbor()

    def get_lower_neighbors(self) -> dict[Point, FilledDiagonal]:
        return self.get_lower_left_neighbor() | self.get_lower_right_neighbor()

    def get_right_neighbors(self) -> dict[Point, FilledDiagonal]:
        return self.get_lower_right_neighbor() | self.get_upper_right_neighbor()

    def get_left_neighbors(self) -> dict[Point, FilledDiagonal]:
        return self.get_lower_left_neighbor() | self.get_upper_left_neighbor()

    def get_upper_left_neighbor(self) -> dict[Point, FilledDiagonal]:
        if self.corner.row > 0 and self.corner.col > 0:
            return {Point(self.corner.row - 1, self.corner.col - 1): Diagonal.BACKSLASH}
        return {}

    def get_upper_right_neighbor(self) -> dict[Point, FilledDiagonal]:
        if self.corner.row > 0 and self.corner.col < self.size:
            return {Point(self.corner.row - 1, self.corner.col): Diagonal.SLASH}
        return {}

    def get_lower_left_neighbor(self) -> dict[Point, FilledDiagonal]:
        if self.corner.row < self.size and self.corner.col > 0:
            return {Point(self.corner.row, self.corner.col - 1): Diagonal.SLASH}
        return {}

    def get_lower_right_neighbor(self) -> dict[Point, FilledDiagonal]:
        if self.corner.row < self.size and self.corner.col < self.size:
            return {Point(self.corner.row, self.corner.col): Diagonal.BACKSLASH}
        return {}

    def notify_value(self, point: Point, value: FilledDiagonal):
        if point not in self.diagonals:
            raise SlantException()
        if self.diagonals[point] == value:
            self.remaining -= 1
            if self.remaining < 0:
                raise SlantException()
        else:
            if self.remaining >= len(self.diagonals):
                raise SlantException()
        del self.diagonals[point]


class Slant:
    def __init__(self,
                 size: int,
                 numbers: dict[Corner, Number],
                 diagonals: list[list[Diagonal]],
                 undetermined_count: Optional[int] = None,
                 corners_connectivity_components: Optional[UnionFind[Corner]] = None) -> None:
        self.size = size
        self.numbers = numbers
        self.diagonals = diagonals
        self.undetermined_count = undetermined_count or size ** 2
        self.corners_connectivity_components = corners_connectivity_components or \
            UnionFind((Corner(i, j) for i in range(size + 1) for j in range(size + 1)))

    def __deepcopy__(self, _) -> Slant:
        return Slant(self.size,
                     deepcopy(self.numbers),
                     deepcopy(self.diagonals),
                     self.undetermined_count,
                     deepcopy(self.corners_connectivity_components),
                     )

    def create_numbers_line(self, line: int) -> str:
        return " ".join((str(self.numbers[Corner(line, j)])
                         if Corner(line, j) in self.numbers else "-"
                         for j in range(self.size + 1)))

    def __str__(self) -> str:
        lines = []
        for i in range(self.size + 1):
            lines.append(self.create_numbers_line(i))
            if i < self.size:
                lines.append(" " + " ".join(map(str, self.diagonals[i])) + " ")

        return "\n".join(lines)

    @staticmethod
    def from_website(website_interface: WebsiteInterface) -> Slant:
        size = website_interface.width
        assert size == website_interface.height

        task = website_interface.task
        cur_board_index = 0
        cur_task_index = 0
        diagonals: list[list[Diagonal]] = [[Diagonal.UNDETERMINED] * size for j in range(size)]
        numbers: dict[Corner, Number] = {}
        while cur_task_index < len(task):
            col = cur_board_index % (size + 1)
            row = cur_board_index // (size + 1)
            corner = Corner(row, col)
            if task[cur_task_index].isdigit():
                numbers[corner] = Number(corner, int(task[cur_task_index]), size)
                cur_board_index += 1
            else:
                cur_board_index += ord(task[cur_task_index]) - 96
            cur_task_index += 1

        assert cur_board_index == (size + 1) ** 2
        return Slant(size, numbers, diagonals)

    def serialize_solution(self) -> str:
        return "".join("".join("f" if c == Diagonal.SLASH else "b" for c in row) for row in self.diagonals)

    def __getitem__(self, point: Point) -> Diagonal:
        return self.diagonals[point.row][point.col]

    def __setitem__(self, point: Point, value: FilledDiagonal):
        if self[point] == value:
            return
        if self[point] != Diagonal.UNDETERMINED:
            raise SlantException()
        corner1, corner2 = point.get_corners(value)
        if self.corners_connectivity_components.find(corner1) is self.corners_connectivity_components.find(corner2):
            raise SlantException()
        self.diagonals[point.row][point.col] = value
        self.undetermined_count -= 1
        for corner in point.get_neighbors():
            if corner in self.numbers:
                self.numbers[corner].notify_value(point, value)
        self.corners_connectivity_components.union(*point.get_corners(value))

    def find_best_number_to_guess(self):
        return max(self.numbers.values(), key=lambda number: (len(number.diagonals) - number.number) if number.diagonals else 0)

    def get_number_corner_count(self, point: Point) -> float:
        if self[point] != Diagonal.UNDETERMINED:
            return -1
        count = 0
        for corner in point.get_corners(Diagonal.BACKSLASH) + point.get_corners(Diagonal.SLASH):
            if corner in self.numbers:
                count += 1
        return count

    def get_point_determined_neighbors(self, point: Point) -> int:
        if self[point] != Diagonal.UNDETERMINED:
            return 10000
        count = 0
        for i in range(-1, 2):
            if point.row + i < 0 or point.row + i >= self.size:
                continue
            for j in range(-1, 2):
                if point.col + j < 0 or point.col + j >= self.size:
                    continue
                if self[Point(point.row + i, point.col + j)] != Diagonal.UNDETERMINED:
                    count += 1
        return count

    def find_best_pos_to_guess(self):
        return max((Point(i, j) for i, j in product(range(self.size), repeat=2)), key=lambda p: (self.get_number_corner_count(p), -self.get_point_determined_neighbors(p)))

    def guess(self):
        guess_pos = self.find_best_pos_to_guess()
        try:
            new_board = deepcopy(self)
            new_board[guess_pos] = Diagonal.SLASH
            new_board.solve()
            self.diagonals = new_board.diagonals
            self.undetermined_count = new_board.undetermined_count
        except SlantException:
            self[guess_pos] = ~Diagonal.SLASH

    def solve(self):
        if self.undetermined_count == self.size ** 2:
            self.bootstrap_start()

        last_undetermined = self.undetermined_count
        while self.undetermined_count > 0:
            for corner, number in self.numbers.items():
                if not number.diagonals:
                    continue
                if number.remaining == 0:
                    for point, diagonal in list(number.diagonals.items()):
                        self[point] = ~diagonal
                elif number.remaining == len(number.diagonals):
                    for point, diagonal in list(number.diagonals.items()):
                        self[point] = diagonal
                else:
                    self.check_rows(corner, number)
                    self.check_cols(corner, number)

            self.search_For_loops()
            if last_undetermined == self.undetermined_count:
                self.guess()
            last_undetermined = self.undetermined_count

    def search_For_loops(self):
        for i in range(self.size):
            for j in range(self.size):
                point = Point(i, j)
                if self[point] != Diagonal.UNDETERMINED:
                    continue
                for diagonal in (Diagonal.BACKSLASH, Diagonal.SLASH):
                    corner1, corner2 = point.get_corners(diagonal)
                    if self.corners_connectivity_components.are_in_the_same_component(corner1, corner2):
                        self[point] = ~diagonal
                        break

    def bootstrap_start(self):
        for corner, number in self.numbers.items():
            if number.number == 1 and 0 < corner.row < self.size - 1 and 0 < corner.col < self.size:
                if corner.col > 1 and Corner(corner.row + 1, corner.col - 1) in self.numbers \
                        and self.numbers[Corner(corner.row + 1, corner.col - 1)].number == 1:
                    self[Point(corner.row, corner.col - 1)] = Diagonal.BACKSLASH
                if corner.col < self.size - 1 and Corner(corner.row + 1, corner.col + 1) in self.numbers \
                        and self.numbers[Corner(corner.row + 1, corner.col + 1)].number == 1:
                    self[Point(corner.row, corner.col)] = Diagonal.SLASH

    def check_cols(self, corner: Corner, number: Number):
        cur_corner = Corner(corner.row, corner.col + 1)
        total_number = number.remaining
        all_neighbors = set(number.diagonals)
        edge_neighbors = number.diagonals.copy()
        while cur_corner in self.numbers:
            cur_number = self.numbers[cur_corner]
            total_number += cur_number.remaining
            all_neighbors |= set(cur_number.diagonals)
            edge_neighbors = dict_symmetric_difference(edge_neighbors, cur_number.diagonals)
            if len(all_neighbors) == total_number:
                for point, diagonal in edge_neighbors.items():
                    self[point] = diagonal
            if len(all_neighbors) - len(edge_neighbors) == total_number:
                for point, diagonal in edge_neighbors.items():
                    self[point] = ~diagonal
            cur_corner = Corner(cur_corner.row, cur_corner.col + 1)
            break

    def check_rows(self, corner: Corner, number: Number):
        cur_corner = Corner(corner.row + 1, corner.col)
        total_number = number.remaining
        all_neighbors = set(number.diagonals)
        edge_neighbors = number.diagonals.copy()
        while cur_corner in self.numbers:
            cur_number = self.numbers[cur_corner]
            total_number += cur_number.remaining
            all_neighbors |= set(cur_number.diagonals)
            edge_neighbors = dict_symmetric_difference(edge_neighbors, cur_number.diagonals)
            if len(all_neighbors) == total_number:
                for point, diagonal in edge_neighbors.items():
                    self[point] = diagonal
            if len(all_neighbors) - len(edge_neighbors) == total_number:
                for point, diagonal in edge_neighbors.items():
                    self[point] = ~diagonal
            cur_corner = Corner(cur_corner.row + 1, cur_corner.col)
            break


def main():
    website_interface = WebsiteInterface(
        "https://www.puzzle-slant.com/", {"size": 10, "specific": 0, "specid": 0})
    board = Slant.from_website(website_interface)
    try:
        board.solve()
        website_interface.submit_solution(board.serialize_solution())
    except:
        print(f"Failed to solve puzzle #{website_interface.puzzle_id}")
        print(board)
        raise


if __name__ == "__main__":
    main()
