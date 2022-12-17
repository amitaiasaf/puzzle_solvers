from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from typing_extensions import Self
from utils.website_interface import WebsiteInterface


class LITSException(Exception):
    pass


@dataclass(frozen=True)
class Point:
    row: int
    col: int

    def get_neighbors(self) -> Points:
        return {Point(self.row - 1, self.col),
                Point(self.row + 1, self.col),
                Point(self.row, self.col - 1),
                Point(self.row, self.col + 1), }

    def __add__(self, other: Point) -> Point:
        return Point(self.row + other.row, self.col + other.col)

    def __deepcopy__(self, _) -> Self:
        return self


class Cell(Enum):
    UNDETERMINED = " "
    MARKED = "#"
    NOT_MARKED = ","

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        return self is other


PuzzleDataType = list[list[Cell]]


Points = set[Point]
FrozenPoints = frozenset[Point]


class ShapeType(Enum):
    L = auto()
    I = auto()
    T = auto()
    S = auto()


L_OPTIONS = [
    {Point(0, 0), Point(1, 0), Point(2, 0), Point(2, 1)},
    {Point(0, 0), Point(1, 0), Point(2, 0), Point(2, -1)},
    {Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1)},
    {Point(0, 0), Point(1, 0), Point(2, 0), Point(0, -1)},
    {Point(0, 0), Point(0, 1), Point(0, 2), Point(1, 2)},
    {Point(0, 0), Point(0, 1), Point(0, 2), Point(-1, 2)},
    {Point(0, 0), Point(0, 1), Point(0, 2), Point(1, 0)},
    {Point(0, 0), Point(0, 1), Point(0, 2), Point(-1, 0)},
]

I_OPTIONS = [
    {Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)},
    {Point(0, 0), Point(0, 1), Point(0, 2), Point(0, 3)},
]

T_OPTIONS = [
    {Point(0, 0), Point(1, 0), Point(2, 0), Point(1, 1)},
    {Point(0, 0), Point(1, 0), Point(2, 0), Point(1, -1)},
    {Point(0, 0), Point(0, 1), Point(0, 2), Point(1, 1)},
    {Point(0, 0), Point(0, 1), Point(0, 2), Point(-1, 1)},
]

S_OPTIONS = [
    {Point(0, 0), Point(1, 0), Point(1, 1), Point(2, 1)},
    {Point(0, 0), Point(1, 0), Point(1, -1), Point(2, -1)},
    {Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 2)},
    {Point(0, 0), Point(0, 1), Point(-1, 1), Point(-1, 2)},
]


@dataclass(frozen=True)
class Shape:
    points: FrozenPoints
    type: ShapeType

    @staticmethod
    def get_l_options(points: Points) -> set[Shape]:
        options = set()
        for pivot in points:
            for option in L_OPTIONS:
                new_shape = frozenset({pivot + p for p in option})
                if new_shape <= points:
                    options.add(Shape(new_shape, ShapeType.L))
        return options

    @staticmethod
    def get_i_options(points: Points) -> set[Shape]:
        options = set()
        for pivot in points:
            for option in I_OPTIONS:
                new_shape = frozenset({pivot + p for p in option})
                if new_shape <= points:
                    options.add(Shape(new_shape, ShapeType.I))
        return options

    @staticmethod
    def get_t_options(points: Points) -> set[Shape]:
        options = set()
        for pivot in points:
            for option in T_OPTIONS:
                new_shape = frozenset({pivot + p for p in option})
                if new_shape <= points:
                    options.add(Shape(new_shape, ShapeType.T))
        return options

    @staticmethod
    def get_s_options(points: Points) -> set[Shape]:
        options = set()
        for pivot in points:
            for option in S_OPTIONS:
                new_shape = frozenset({pivot + p for p in option})
                if new_shape <= points:
                    options.add(Shape(new_shape, ShapeType.S))
        return options

    @staticmethod
    def get_all_options(points: Points) -> set[Shape]:
        options = Shape.get_l_options(points)
        options.update(Shape.get_i_options(points))
        options.update(Shape.get_t_options(points))
        options.update(Shape.get_s_options(points))
        return options

    def __deepcopy__(self, _) -> Self:
        return self


class Block:
    def __init__(self, points: Points,
                 orig_points: Optional[Points] = None,
                 options: Optional[set[Shape]] = None,
                 marked: bool = False) -> None:
        self.points = points
        if orig_points is not None:
            self.orig_points = orig_points
        else:
            self.orig_points = self.points.copy()
        if options is None:
            self.options = Shape.get_all_options(points)
        else:
            self.options = options
        self.marked = marked

    def __deepcopy__(self, _) -> Self:
        return Block(self.points.copy(), self.orig_points, self.options.copy(), self.marked)

    def set_point_unmarked(self, point: Point):
        if point not in self.points:
            return
        self.points.remove(point)
        for option in list(self.options):
            if point in option.points:
                self.options.remove(option)
        if not self.options:
            raise LITSException()

    def set_point_marked(self, point: Point):
        if point not in self.points:
            return
        self.points.remove(point)
        for option in list(self.options):
            if point not in option.points:
                self.options.remove(option)
        if not self.options:
            raise LITSException()

    def get_marked_points(self) -> Points:
        possible_points = self.points.copy()
        for option in self.options:
            possible_points.intersection_update(option.points)
            if not possible_points:
                return set()
        return possible_points

    def get_unmarked_points(self) -> Points:
        possible_points = self.points.copy()
        for option in self.options:
            possible_points.difference_update(option.points)
            if not possible_points:
                return set()
        return possible_points

    def remove_options_that_form_2_on_2_blocks(self, board_points: Points):
        for option in list(self.options):
            cur_points = board_points | option.points
            for point in option.points:
                if {point + Point(1, 0), point + Point(0, 1), point + Point(1, 1)} <= cur_points:
                    self.options.remove(option)
                    break
                if {point + Point(1, 0), point + Point(0, -1), point + Point(1, -1)} <= cur_points:
                    self.options.remove(option)
                    break
                if {point + Point(-1, 0), point + Point(0, 1), point + Point(-1, 1)} <= cur_points:
                    self.options.remove(option)
                    break
                if {point + Point(-1, 0), point + Point(0, -1), point + Point(-1, -1)} <= cur_points:
                    self.options.remove(option)
                    break
        if not self.options:
            raise LITSException()

    def remove_options_by_point_and_shape(self, point: Point, shape_type: ShapeType):
        for option in list(self.options):
            if shape_type == option.type and point in option.points:
                self.options.remove(option)
        if not self.options:
            raise LITSException()


class LITS:
    def __init__(self,
                 size: int,
                 blocks: list[Block],
                 cells: Optional[PuzzleDataType] = None,
                 undetermined_count: int = 0,
                 marked_points: Optional[Points] = None) -> None:

        self.size = size
        self.blocks = blocks
        self.cells = cells if cells is not None else [[Cell.UNDETERMINED for i in range(size)] for j in range(size)]
        self.cells_to_blocks = self.create_cells_to_blocks_mapping()
        self.undetermined_count = undetermined_count or self.size ** 2
        self.marked_points: Points = marked_points if marked_points is not None else set()

    def __deepcopy__(self, _) -> LITS:
        return LITS(self.size,
                    deepcopy(self.blocks),
                    deepcopy(self.cells),
                    self.undetermined_count,
                    self.marked_points.copy())

    def __str__(self) -> str:
        lines = []
        lines.append(" - ".join(" " * (self.size + 1)))
        for i, line in enumerate(self.cells):
            next_line_data = [" "]
            cur_line_data = ["|"]
            for j, cell in enumerate(line):
                cur_line_data.append(str(cell))
                if j == self.size - 1 or self.cells_to_blocks[Point(i, j)] is not self.cells_to_blocks[Point(i, j + 1)]:
                    cur_line_data.append("|")
                else:
                    cur_line_data.append(" ")
                if i == self.size - 1 or self.cells_to_blocks[Point(i, j)] is not self.cells_to_blocks[Point(i + 1, j)]:
                    next_line_data.append("-")
                else:
                    next_line_data.append(" ")
                next_line_data.append(" ")
            lines.append(" ".join(cur_line_data))
            lines.append(" ".join(next_line_data))
        return "\n".join(lines)

    def serialize_solution(self) -> str:
        return "".join("".join("y" if c == Cell.MARKED else "n" for c in row) for row in self.cells)

    def __getitem__(self, point: Point) -> Cell:
        return self.cells[point.row][point.col]

    @staticmethod
    def from_website(website_interface: WebsiteInterface):
        assert website_interface.width == website_interface.height
        size = website_interface.width
        task = website_interface.task
        block_ids = list(map(int, task.split(",")))
        block_points: dict[int, set[Point]] = {}
        for i, block_id in enumerate(block_ids):
            row = i // size
            col = i % size
            if block_id not in block_points:
                block_points[block_id] = set()
            block_points[block_id].add(Point(row, col))

        blocks = [Block(points) for points in block_points.values()]
        return LITS(size, blocks)

    def create_cells_to_blocks_mapping(self) -> dict[Point, Block]:
        cells_to_blocks: dict[Point, Block] = {}
        for block in self.blocks:
            for point in block.orig_points:
                cells_to_blocks[point] = block

        return cells_to_blocks

    def set_cell_as_unmarked(self, point: Point):
        if self[point] == Cell.NOT_MARKED:
            return
        if self[point] == Cell.MARKED:
            raise LITSException()
        self.cells[point.row][point.col] = Cell.NOT_MARKED
        self.cells_to_blocks[point].set_point_unmarked(point)
        self.undetermined_count -= 1

    def set_cell_as_marked(self, point: Point):
        if self[point] == Cell.MARKED:
            return
        if self[point] == Cell.NOT_MARKED:
            raise LITSException()
        self.cells[point.row][point.col] = Cell.MARKED
        self.cells_to_blocks[point].set_point_marked(point)
        self.undetermined_count -= 1
        self.marked_points.add(point)
        if not self.cells_to_blocks[point].points:
            self.mark_block(self.cells_to_blocks[point])

    def mark_block(self, block: Block):
        if block.marked:
            return
        block.marked = True
        shape = next(iter(block.options))
        for point in block.points - shape.points:
            self.set_cell_as_unmarked(point)
        for point in shape.points:
            for neighbor in point.get_neighbors():
                if neighbor.row < 0 or neighbor.col < 0 or neighbor.row >= self.size or neighbor.col >= self.size:
                    continue
                if self.cells_to_blocks[point] is self.cells_to_blocks[neighbor]:
                    continue
                if self[neighbor] == Cell.NOT_MARKED:
                    continue
                self.cells_to_blocks[neighbor].remove_options_by_point_and_shape(neighbor, shape.type)
        block.remove_options_that_form_2_on_2_blocks(self.marked_points)

    def get_connected_component(self, point: Point) -> Points:
        component = set()
        pending = {point}
        while pending:
            p = pending.pop()
            for neighbor in p.get_neighbors():
                if neighbor not in self.marked_points or neighbor in component:
                    continue
                pending.add(neighbor)
            component.add(p)
        return component

    def get_undetermined_neighbors(self, component: Points) -> Points:
        undetermined: Points = set()
        for p in component:
            for neighbor in p.get_neighbors():
                if neighbor.row < 0 or neighbor.col < 0 or neighbor.row >= self.size or neighbor.col >= self.size:
                    continue
                if self[neighbor] == Cell.UNDETERMINED:
                    undetermined.add(neighbor)
        return undetermined

    def find_best_block_to_guess(self):
        return min(range(len(self.blocks)), key=lambda i: (len(self.blocks[i].options), i) if len(self.blocks[i].options) > 1 else (10000, 10000))

    def guess(self):
        guess_block = self.find_best_block_to_guess()
        guess = next(iter(self.blocks[guess_block].options))
        try:
            new_board = deepcopy(self)
            new_board.blocks[guess_block].options = {guess}
            new_board.solve()
            self.cells = new_board.cells
            self.blocks = new_board.blocks
            self.marked_points = new_board.marked_points
            self.undetermined_count = new_board.undetermined_count
        except LITSException:
            self.blocks[guess_block].options.remove(guess)

    def solve(self):
        last_undetermined = self.undetermined_count
        while self.undetermined_count:
            change_in_options = False
            for block in self.blocks:
                if not block.points:
                    if len(block.options) == 1:
                        self.mark_block(block)
                    else:
                        raise LITSException()
                if len(block.options) > 1:
                    block.remove_options_that_form_2_on_2_blocks(self.marked_points)
                for point in list(block.get_marked_points()):
                    self.set_cell_as_marked(point)
                for point in list(block.get_unmarked_points()):
                    self.set_cell_as_unmarked(point)
                if not block.options:
                    raise LITSException()

            passed_points = set()
            for p in list(self.marked_points):
                if p in passed_points:
                    continue
                component = self.get_connected_component(p)
                undetermined_neighbors = self.get_undetermined_neighbors(component)
                if len(component) == 4 * len(self.blocks):
                    break
                if len(undetermined_neighbors) == 0:
                    raise LITSException()
                elif len(undetermined_neighbors) == 1:
                    self.set_cell_as_marked(undetermined_neighbors.pop())
                passed_points.update(component)

            if last_undetermined == self.undetermined_count and not change_in_options:
                self.guess()
            last_undetermined = self.undetermined_count
        for block in self.blocks:
            self.mark_block(block)
        for block in self.blocks:
            if not block.options:
                raise LITSException()


def main():
    website_interface = WebsiteInterface(
        "https://www.puzzle-lits.com/", {"size": 12, "specific": 0, "specid": "123456"})
    board = LITS.from_website(website_interface)
    try:
        board.solve()
        website_interface.submit_solution(board.serialize_solution())
    except:
        print(f"Failed to solve puzzle #{website_interface.puzzle_id}")
        print(board)
        raise


if __name__ == "__main__":
    main()
