from __future__ import annotations
import copy
from dataclasses import dataclass
from enum import Enum
import random
import time
from typing import Counter, Iterable, List, Set, Tuple


from utils.website_interface import WebsiteInterface


class NurikabeException(Exception):
    pass


class CellType(Enum):
    UNDETERMINED = -1
    BLACK = 0
    WHITE = 1
    NUMBER = 2
    IGNORED = 3


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
        elif self.type == CellType.IGNORED:
            return "*"


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

    def get_neighbors(self, puzzle_data: PuzzleDataType) -> Points:
        points = set()
        if self.row > 0:
            points.add(Point(self.row - 1, self.col))
        if self.col > 0:
            points.add(Point(self.row, self.col - 1))
        if self.row + 1 < len(puzzle_data):
            points.add(Point(self.row + 1, self.col))
        if self.col + 1 < len(puzzle_data[0]):
            points.add(Point(self.row, self.col + 1))
        return Points(points)


class Points(Set[Point]):
    def get_neighbors(self, board: Nurikabe) -> Points:
        neighbors = Points()
        for point in self:
            neighbors.update(point.get_neighbors(board.puzzle_data))
        neighbors.difference_update(self)
        return neighbors

    def get_non_black_neighbors(self, board: Nurikabe) -> Points:
        neighbors: Points = self.get_neighbors(board)
        neighbors.difference_update(board.black_places)
        neighbors.difference_update(board.ignored_places)
        return neighbors

    def get_black_neighbors(self, board: Nurikabe) -> Points:
        neighbors: Points = self.get_neighbors(board)
        neighbors.intersection_update(board.black_places)
        return neighbors

    def get_lonely_whites_neighbors(self, board: Nurikabe) -> Points:
        neighbors: Points = self.get_neighbors(board)
        neighbors.intersection_update(board.lonely_whites)
        return neighbors

    def get_undetermined_neighbors(self, board: Nurikabe) -> Points:
        neighbors: Points = self.get_neighbors(board)
        neighbors.intersection_update(board.undetermined_places)
        return neighbors

    def copy(self):
        return self.__class__(self)


class NumberArea(Points):
    def __init__(self, points: Points, number: int) -> None:
        super().__init__(points)
        self.number = number
        self.filled_neighbors = False

    def copy(self) -> None:
        other = self.__class__(self, self.number)
        other.filled_neighbors = self.filled_neighbors
        return other

    def is_complete(self) -> bool:
        return len(self) == self.number

    def add_point(self, point: Point, board: Nurikabe):
        if board[point].type in (CellType.BLACK, CellType.NUMBER, CellType.IGNORED):
            raise NurikabeException()
        self.add(point)
        if len(self) > self.number:
            raise NurikabeException()
        board.mark_white(point)
        board.lonely_whites.discard(point)

    def add_points(self, points: Points, board: Nurikabe):
        for point in points:
            self.add_point(point, board)

    def get_reachable_points(self, board: Nurikabe):
        cur = self
        for _ in range(self.number - len(self)):
            cur = Points(cur | cur.get_non_black_neighbors(board))
        return Points(cur - self)

    def get_distance(self, other: Points, board: Nurikabe) -> int:
        cur = self
        for i in range(self.number - len(self)):
            cur = Points(cur | cur.get_non_black_neighbors(board))
            if cur & other:
                return i
        return 10000


class Nurikabe:
    def __init__(self, width: int, height: int, puzzle_data: PuzzleDataType) -> None:
        self.width = width
        self.height = height
        self.puzzle_data = puzzle_data
        self.number_areas: List[NumberArea] = []
        self.ignored_places = Points()
        self.black_places = Points()
        self.lonely_whites = Points()
        self.undetermined_places = Points()

        for row in range(self.height):
            for col in range(self.width):
                if self[Point(row, col)].type == CellType.IGNORED:
                    self.ignored_places.add(Point(row, col))
                elif self[Point(row, col)].type == CellType.NUMBER:
                    self.number_areas.append(NumberArea({Point(row, col)}, self[Point(row, col)].number))
                elif self[Point(row, col)].type == CellType.UNDETERMINED:
                    self.undetermined_places.add(Point(row, col))

    def __str__(self) -> str:
        return "\n".join(("\t".join((str(c) for c in row)) for row in self.puzzle_data))

    def __getitem__(self, point: Point) -> Cell:
        return self.puzzle_data[point.row][point.col]

    def mark_black(self, points: Points):
        for point in points:
            if self[point].type in (CellType.BLACK, CellType.IGNORED):
                continue
            if self[point].type in (CellType.WHITE, CellType.NUMBER):
                raise NurikabeException()
            self[point].type = CellType.BLACK
            self.black_places.add(point)
            self.undetermined_places.remove(point)

    def mark_white_allow_ignored(self, point: Point):
        if self[point].type == CellType.BLACK:
            raise NurikabeException()
        if self[point].type != CellType.UNDETERMINED:
            return
        self[point].type = CellType.WHITE
        self.undetermined_places.discard(point)
        self.lonely_whites.add(point)

    def mark_white(self, point: Point):
        if self[point].type in (CellType.BLACK, CellType.IGNORED):
            raise NurikabeException()
        self.mark_white_allow_ignored(point)

    def fill_complete_number_area_neighbors(self, number_area: NumberArea):
        neighbors = number_area.get_non_black_neighbors(self)
        self.mark_black(neighbors)
        number_area.filled_neighbors = True

    def avoid_2x2_blacks(self):
        for i in range(self.height):
            for j in range(self.width - 1):
                if self[Point(i, j)].type == CellType.BLACK and self[Point(i, j+1)].type == CellType.BLACK:
                    if i > 0:
                        if self[Point(i - 1, j)].type == CellType.BLACK:
                            self.mark_white_allow_ignored(Point(i - 1, j + 1))
                        if self[Point(i - 1, j + 1)].type == CellType.BLACK:
                            self.mark_white_allow_ignored(Point(i - 1, j))
                    if i + 1 < self.height:
                        if self[Point(i + 1, j)].type == CellType.BLACK:
                            self.mark_white_allow_ignored(Point(i + 1, j + 1))
                        if self[Point(i + 1, j + 1)].type == CellType.BLACK:
                            self.mark_white_allow_ignored(Point(i + 1, j))

    def get_black_component(self, point: Point) -> Points:
        component = Points({point})
        while True:
            neighbors = component.get_black_neighbors(self)
            if not neighbors:
                return component
            component.update(neighbors)

    def avoid_isolated_blacks(self):
        remaining_blacks = self.black_places.copy()
        while remaining_blacks:
            component = self.get_black_component(remaining_blacks.pop())
            remaining_blacks.difference_update(component)
            if not remaining_blacks:
                break
            undetermined_neighbors = component.get_undetermined_neighbors(self)
            if len(undetermined_neighbors) == 1:
                self.mark_black(undetermined_neighbors)
            if not undetermined_neighbors:
                raise NurikabeException()

    def get_lonely_white_component(self, point: Point) -> Points:
        component = Points({point})
        while True:
            neighbors = component.get_lonely_whites_neighbors(self)
            if not neighbors:
                return component
            component.update(neighbors)

    def avoid_isolated_whites(self):
        remaining_lonely_whites = self.lonely_whites.copy()
        while remaining_lonely_whites:
            component = self.get_lonely_white_component(remaining_lonely_whites.pop())
            remaining_lonely_whites.difference_update(component)
            neighbors = component.get_non_black_neighbors(self)
            if not neighbors:
                raise NurikabeException()
            for neighbor in neighbors:
                if self[neighbor].type == CellType.NUMBER:
                    break
            else:
                if len(neighbors) == 1:
                    self.mark_white(neighbors.pop())
            for number_area in self.number_areas:
                if number_area.is_complete():
                    continue
                if number_area.get_distance(component, self) + len(component) + len(number_area) <= number_area.number:
                    break
            else:
                raise NurikabeException()

    def copy(self) -> Nurikabe:
        new_board = Nurikabe(self.width, self.height, copy.deepcopy(self.puzzle_data))
        new_board.lonely_whites = self.lonely_whites.copy()
        new_board.black_places = self.black_places.copy()
        new_board.number_areas = [number_area.copy() for number_area in self.number_areas]
        return new_board

    def get_best_number_area_to_guess(self) -> NumberArea:
        return min(self.number_areas, key=lambda na: (na.number - len(na) if not na.is_complete() else 1000))

    def get_guess(self) -> Point:
        if self.lonely_whites:
            return self.lonely_whites.get_undetermined_neighbors(self).pop()
        else:
            # number_area = self.get_best_number_area_to_guess()
            number_area = random.choice(self.number_areas)
            while number_area.is_complete():
                number_area = random.choice(self.number_areas)
            return number_area.get_undetermined_neighbors(self).pop()

    def guess(self) -> None:
        point_to_guess = self.get_guess()
        new_board = self.copy()
        new_board.mark_white(point_to_guess)
        try:
            new_board.solve()
            self.puzzle_data = new_board.puzzle_data
            self.undetermined_places = new_board.undetermined_places
        except NurikabeException:
            self.mark_black({point_to_guess})

    def solve(self) -> None:
        remaining = 100000
        prev_remaining = 100000
        while prev_remaining:
            change_not_in_undetermined = False
            self.avoid_2x2_blacks()
            self.avoid_isolated_blacks()
            self.avoid_isolated_whites()

            neighbors_counter = Counter()
            reachable = Points()
            for number_area in self.number_areas:
                if not number_area.is_complete():
                    neighbors = number_area.get_non_black_neighbors(self)
                    reachable |= number_area.get_reachable_points(self)
                    if len(neighbors) == 0:
                        raise NurikabeException()
                    if neighbors & self.lonely_whites:
                        for point in neighbors & self.lonely_whites:
                            number_area.add_points(self.get_lonely_white_component(point), self)
                        change_not_in_undetermined = True
                    if len(neighbors) == 1:
                        number_area.add_points(neighbors, self)
                        change_not_in_undetermined = True
                    if len(reachable) + len(number_area) == number_area.number:
                        number_area.add_points(reachable, self)
                        change_not_in_undetermined = True
                    neighbors_counter.update(neighbors)

                if number_area.is_complete():
                    if not number_area.filled_neighbors:
                        self.fill_complete_number_area_neighbors(number_area)
            self.mark_black(self.undetermined_places - reachable)
            if self.lonely_whites - reachable:
                raise NurikabeException()
            for point, count in neighbors_counter.items():
                if count > 1:
                    self.mark_black(Points({point}))

            if remaining and len(self.undetermined_places) == remaining and not change_not_in_undetermined:
                self.guess()
                if not self.undetermined_places:
                    break
            prev_remaining = remaining
            remaining = len(self.undetermined_places)

    def serialize_solution(self):
        return "".join("".join(["y" if c.type == CellType.BLACK else "n" for c in row]) for row in self.puzzle_data)


def init_puzzle_from_website(website_interface: WebsiteInterface) -> Nurikabe:
    width = website_interface.width
    height = website_interface.height
    task = website_interface.task
    cur_board_index = 0
    cur_task_index = 0
    puzzle_data: PuzzleDataType = [[Cell() for i in range(width)] for j in range(height)]
    while cur_task_index < len(task):
        col = cur_board_index % width
        row = cur_board_index // width
        if task[cur_task_index] == '_':
            cur_task_index += 1
        elif task[cur_task_index] == 'B':
            puzzle_data[row][col] = Cell(CellType.IGNORED)
            cur_board_index += 1
            cur_task_index += 1
        elif task[cur_task_index].isdigit():
            end = cur_task_index
            while True:
                end += 1
                if end >= len(task) or not task[end].isdigit():
                    break
            puzzle_data[row][col] = NumberCell(number=int(task[cur_task_index:end]))
            cur_board_index += 1
            cur_task_index = end
        else:
            cur_board_index += ord(task[cur_task_index]) - 96
            cur_task_index += 1

    assert cur_board_index == width * height
    return Nurikabe(width, height, puzzle_data)


def main():
    website_interface = WebsiteInterface("https://www.puzzle-nurikabe.com/?size=3")
    board = init_puzzle_from_website(website_interface)
    board.solve()
    website_interface.submit_solution(board.serialize_solution())


if __name__ == "__main__":
    main()
