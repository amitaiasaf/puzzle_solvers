from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Collection

from utils.website_interface import WebsiteInterface


class ThermometersException(Exception):
    pass


class Cell(Enum):
    UNDETERMINED = " "
    LIQUID = "X"
    AIR = ","


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    DOWN = 3
    UP = 4
    CURVED = 5


@dataclass(frozen=True)
class Point:
    row: int
    col: int


Points = list[Point]


@dataclass
class Thermometer:
    points: Points
    liquid: int = 0
    air: int = 0

    def get_points_by_row(self, row: int) -> Points:
        return list(filter(lambda p: p.row == row, self.points))

    def get_points_by_col(self, col: int) -> Points:
        return list(filter(lambda p: p.col == col, self.points))


@dataclass
class Line:
    line_id: int
    is_col: bool
    required_liquid: int
    cur_liquid: int = 0
    cur_air: int = 0


class Thermometers:
    def __init__(self,
                 size: int,
                 rows: list[Line],
                 cols: list[Line],
                 thermometers: list[Thermometer],
                 cell_to_thermometer: dict[Point, int]) -> None:
        self.size = size
        self.rows = rows
        self.cols = cols
        self.thermometers = thermometers
        self.cell_to_thermometer = cell_to_thermometer
        self.undetermined_count = size ** 2
        self.cells = [[Cell.UNDETERMINED for _ in range(size)] for _ in range(size)]

    @staticmethod
    def from_website(website_interface: WebsiteInterface) -> Thermometers:
        assert website_interface.width == website_interface.height
        size = website_interface.width
        task = website_interface.task
        side, thermometers_data = task.split(";", maxsplit=1)
        side_numbers = list(map(int, side.split("_")))

        cell_to_thermometer = dict[Point, int]()

        cols = [Line(i, True, number) for i, number in enumerate(side_numbers[:size])]
        rows = [Line(i, False, number) for i, number in enumerate(side_numbers[size:])]

        next_cell = Point(0, 0)

        thermometers = list[Thermometer]()
        for i, thermometer_data in enumerate(thermometers_data.split(";")):
            next_cell = Thermometers.get_next_cell(size, cell_to_thermometer, next_cell)
            direction, data = thermometer_data.split(",", maxsplit=1)
            match Direction(int(direction)):
                case Direction.RIGHT:
                    points = [Point(next_cell.row, next_cell.col + i) for i in range(int(data))]
                case Direction.LEFT:
                    points = [Point(next_cell.row, next_cell.col + i) for i in range(int(data) - 1, -1, -1)]
                case Direction.DOWN:
                    points = [Point(next_cell.row + i, next_cell.col) for i in range(int(data))]
                case Direction.UP:
                    points = [Point(next_cell.row + i, next_cell.col) for i in range(int(data) - 1, -1, -1)]
                case Direction.CURVED:
                    points = [Point(d // size, d % size) for d in map(int, data.split(","))]
            assert cell_to_thermometer.keys().isdisjoint(points)
            for point in points:
                cell_to_thermometer[point] = i
            thermometers.append(Thermometer(points))

        return Thermometers(size, rows, cols, thermometers, cell_to_thermometer)

    @staticmethod
    def get_next_cell(size, cells_to_thermometers: Collection[Point], next_cell: Point):
        for row in range(next_cell.row, size):
            for col in range(size):
                if Point(row, col) not in cells_to_thermometers:
                    return Point(row, col)
        raise RuntimeError("All cells are used")

    def serialize_solution(self) -> str:
        return "".join("".join("y" if c == Cell.LIQUID else "n" for c in row) for row in self.cells)

    def __getitem__(self, point: Point) -> Cell:
        return self.cells[point.row][point.col]

    def __setitem__(self, point: Point, value: Cell):
        self.cells[point.row][point.col] = value

    def mark_liquid(self, point: Point):
        for p in self.thermometers[self.cell_to_thermometer[point]].points:
            self.mark_point_as_liquid(p)
            if p == point:
                break

    def mark_point_as_liquid(self, point: Point):
        if self[point] == Cell.LIQUID:
            return
        if self[point] == Cell.AIR:
            raise ThermometersException()
        self[point] = Cell.LIQUID
        self.undetermined_count -= 1
        self.rows[point.row].cur_liquid += 1
        self.cols[point.col].cur_liquid += 1

    def mark_air(self, point: Point):
        for p in self.thermometers[self.cell_to_thermometer[point]].points[::-1]:
            self.mark_point_as_air(p)
            if p == point:
                break

    def mark_point_as_air(self, point: Point):
        if self[point] == Cell.AIR:
            return
        if self[point] == Cell.LIQUID:
            raise ThermometersException()
        self[point] = Cell.AIR
        self.undetermined_count -= 1
        self.rows[point.row].cur_air += 1
        self.cols[point.col].cur_air += 1

    def get_intersection_with_line(self, thermometer: Thermometer, line: Line) -> Points:
        if line.is_col:
            return [p for p in thermometer.points if p.col == line.line_id and self[p] == Cell.UNDETERMINED]
        return [p for p in thermometer.points if p.row == line.line_id and self[p] == Cell.UNDETERMINED]

    def get_intersection_with_col(self, thermometer: Thermometer, col: Line) -> Points:
        return [p for p in thermometer.points if p.col == col.line_id and self[p] == Cell.UNDETERMINED]

    def solve(self):
        last_undetermined = self.undetermined_count
        while self.undetermined_count:
            for line in self.rows + self.cols:
                for thermometer in self.thermometers:
                    intersection = self.get_intersection_with_line(thermometer, line)
                    if not intersection:
                        continue
                    self.fill_thermometer_with_liquid(line, intersection)
                    intersection = self.get_intersection_with_line(thermometer, line)
                    if not intersection:
                        continue
                    self.fill_thermometer_with_air(line, intersection)
            if last_undetermined == self.undetermined_count:
                raise ThermometersException()
            last_undetermined = self.undetermined_count

    def fill_thermometer_with_liquid(self, line: Line, intersection: Points):
        additional_liquid = line.required_liquid - (self.size - line.cur_air) + len(intersection)
        if additional_liquid > 0:
            self.mark_liquid(intersection[additional_liquid - 1])

    def fill_thermometer_with_air(self, line: Line, intersection: Points):
        additional_air = line.cur_liquid - line.required_liquid + len(intersection)
        if additional_air > 0:
            self.mark_air(intersection[-additional_air])


def main():
    website_interface = WebsiteInterface(
        "https://www.puzzle-thermometers.com/", {"size": 13, "specific": 0, "specid": 13112563})
    board = Thermometers.from_website(website_interface)
    board.solve()
    website_interface.submit_solution(board.serialize_solution())


if __name__ == "__main__":
    main()
