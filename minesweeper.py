from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Set
import time

import requests


class MinesweeperException(Exception):
    pass


class CellType(Enum):
    UNDETERMINED = -1
    MINE = 0
    SAFE = 1
    NUMBER = 2


@dataclass
class Cell:
    type: CellType = CellType.UNDETERMINED

    def __str__(self) -> str:
        if self.type == CellType.UNDETERMINED:
            return ","
        elif self.type == CellType.MINE:
            return "#"
        elif self.type == CellType.SAFE:
            return "-"


@dataclass
class NumberCell(Cell):
    number: int = 0
    type: CellType = CellType.NUMBER

    def __str__(self) -> str:
        return str(self.number)


PuzzleDataType = List[List[Cell]]


@dataclass(frozen=True)
class Point:
    row: int
    col: int


Points = Set[Point]


@dataclass
class KnownNumberOfMinesArea:
    board: Minesweeper
    points: Points
    number_of_mines: int
    intersections: List[KnownNumberOfMinesArea] = field(default_factory=list)

    @staticmethod
    def create_from_number_cell(board: Minesweeper, point: Point) -> KnownNumberOfMinesArea:
        assert board[point].type == CellType.NUMBER

        start_row = max(0, point.row - 1)
        end_row = min(board.height - 1, point.row + 1)
        start_col = max(0, point.col - 1)
        end_col = min(board.width - 1, point.col + 1)

        points: Points = set()
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                if board[Point(row, col)].type != CellType.NUMBER:
                    points.add(Point(row, col))

        number_of_mines = board[point].number

        return KnownNumberOfMinesArea(board, points, number_of_mines)

    def remove_known_points(self) -> None:
        if self.number_of_mines > len(self.points):
            raise MinesweeperException()
        self.number_of_mines -= sum(map(lambda p: self.board[p].type == CellType.MINE, self.points))
        if self.number_of_mines < 0:
            raise MinesweeperException()

        self.points = set(filter(lambda p: self.board[p].type == CellType.UNDETERMINED, self.points))


class Minesweeper:
    def __init__(self, width: int, height: int, puzzle_data: PuzzleDataType) -> None:
        self.width = width
        self.height = height
        self.puzzle_data = puzzle_data

        self.undetermined_count = 0
        self.known_number_of_mines_areas: List[KnownNumberOfMinesArea] = []

        for row in range(self.height):
            for col in range(self.width):
                point = Point(row, col)
                if self[point].type == CellType.UNDETERMINED:
                    self.undetermined_count += 1
                elif self[point].type == CellType.NUMBER:
                    self.known_number_of_mines_areas.append(KnownNumberOfMinesArea.create_from_number_cell(self, point))

        self.init_areas_intersections()

    def init_areas_intersections(self) -> None:
        for i, area in enumerate(self.known_number_of_mines_areas):
            for other_area in self.known_number_of_mines_areas[i + 1:]:
                if area.points & other_area.points:
                    area.intersections.append(other_area)
                    other_area.intersections.append(area)

    def __str__(self) -> str:
        return "\n".join((" ".join((str(c) for c in row)) for row in self.puzzle_data))

    def __getitem__(self, point: Point) -> Cell:
        return self.puzzle_data[point.row][point.col]

    def mark_as_mine(self, point: Point) -> None:
        if self[point].type != CellType.UNDETERMINED:
            raise MinesweeperException()
        self[point].type = CellType.MINE
        self.undetermined_count -= 1

    def mark_all_as_mines(self, points: Point) -> None:
        for point in points:
            self.mark_as_mine(point)

    def mark_as_safe(self, point: Point):
        if self[point].type != CellType.UNDETERMINED:
            raise MinesweeperException()
        self[point].type = CellType.SAFE
        self.undetermined_count -= 1

    def mark_all_as_safe(self, points: Point) -> None:
        for point in points:
            self.mark_as_safe(point)

    def solve(self) -> None:
        while self.undetermined_count > 0:
            last_undetermined = self.undetermined_count
            for area in self.known_number_of_mines_areas:
                area.remove_known_points()

                if area.number_of_mines == 0:
                    self.mark_all_as_safe(area.points)
                    continue
                elif len(area.points) == area.number_of_mines:
                    self.mark_all_as_mines(area.points)
                    continue

                for other_area in area.intersections:
                    other_area.remove_known_points()
                    if other_area.points < area.points:
                        if other_area.number_of_mines > area.number_of_mines:
                            raise MinesweeperException()
                        area.points.difference_update(other_area.points)
                        area.number_of_mines -= other_area.number_of_mines

                    if len(area.points - other_area.points) == area.number_of_mines - other_area.number_of_mines:
                        self.mark_all_as_mines(area.points - other_area.points)
                        self.mark_all_as_safe(other_area.points - area.points)
                        area.remove_known_points()
                        other_area.remove_known_points()

            self.known_number_of_mines_areas = list(filter(lambda area: area.points, self.known_number_of_mines_areas))
            if last_undetermined == self.undetermined_count:
                raise MinesweeperException()


def init_puzzle_from_task(task: str, width: int, height: int) -> Minesweeper:
    cur_board_index = 0
    cur_task_index = 0
    puzzle_data: PuzzleDataType = [[Cell() for i in range(width)] for j in range(height)]
    while cur_task_index < len(task):
        col = cur_board_index % width
        row = cur_board_index // width
        if task[cur_task_index].isdigit():
            puzzle_data[row][col] = NumberCell(number=int(task[cur_task_index]))
            cur_board_index += 1
        else:
            cur_board_index += ord(task[cur_task_index]) - 96
        cur_task_index += 1

    assert cur_board_index == width * height
    return Minesweeper(width, height, puzzle_data)


def load_puzzle_from_internet():
    request = requests.post("https://www.puzzle-minesweeper.com/monthly-minesweeper/")
    text = request.text

    task = re.search(r"var task = '([^']*)'", text).group(1)
    width = int(re.search(r'name="w" value="(\d+)"', text).group(1))
    height = int(re.search(r'name="h" value="(\d+)"', text).group(1))
    return task, width, height


puzzle = load_puzzle_from_internet()
start_time = time.time()
board = init_puzzle_from_task(*puzzle)
print(board)
board.solve()
print(f"Solved board (took {time.time() - start_time} seconds):")
print(board)
