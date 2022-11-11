from __future__ import annotations
from copy import deepcopy
import itertools
from typing import FrozenSet, Iterable, List, Optional, Reversible, Set, Tuple
from utils.website_interface import WebsiteInterface


def powerset(iterable: Iterable, min_size=0) -> Iterable[FrozenSet]:
    return itertools.chain.from_iterable(map(frozenset, itertools.combinations(iterable, i)) for i in range(min_size, len(iterable)))


class SkyscrappersException(Exception):
    pass


class Cell:
    def __init__(self, size: int, number: Optional[int] = None, options: Optional[Set[int]] = None) -> None:
        if number:
            self.number = number
            self.options = {number}
        else:
            self.number = None
            self.options = options.copy() if options else set(range(1, size + 1))

    def __str__(self) -> str:
        if self.number:
            return str(self.number)
        else:
            return "M"

    def __deepcopy__(self, _) -> Cell:
        return Cell(0, self.number, self.options)

    def discard(self, number: int):
        if not self.number:
            self.options.discard(number)
        elif number == self.number:
            raise SkyscrappersException()

    def discard_set(self, numbers: Set[int]):
        for number in numbers:
            self.discard(number)

    def set(self, number: int):
        if number not in self.options:
            raise SkyscrappersException()
        self.number = number
        self.options = {number}

    def reduce_options(self, values: Set[int]):
        self.options.intersection_update(values)
        if not self.options:
            raise SkyscrappersException()


class Line:
    LineOption = Tuple[int]

    def __init__(self, size: int,
                 forward_number: Optional[int],
                 reverse_number: Optional[int],
                 options: Optional[list[LineOption]] = None) -> None:
        self.size = size
        self.forward_number = forward_number
        self.reverse_number = reverse_number
        if options is not None:
            self.options = options
        else:
            self.options = list(filter(self.is_option_compatible, itertools.permutations(range(1, size + 1))))

    def __deepcopy__(self, _) -> Line:
        return Line(self.size, self.forward_number, self.reverse_number, self.options)

    @staticmethod
    def get_forward_number(option: Iterable[int]):
        cur_height = 0
        cur_number = 0
        for height in option:
            if height > cur_height:
                cur_height = height
                cur_number += 1
        return cur_number

    @classmethod
    def get_reverse_number(cls, option: Reversible[int]):
        return cls.get_forward_number(reversed(option))

    def is_option_compatible(self, option: Tuple[int]):
        if self.forward_number and self.get_forward_number(option) != self.forward_number:
            return False
        if self.reverse_number and self.get_reverse_number(option) != self.reverse_number:
            return False
        return True

    def get_possible_values_list(self) -> List[Set[int]]:
        values_list = [set() for i in range(self.size)]
        for option in self.options:
            for i, val in enumerate(option):
                values_list[i].add(val)
        return values_list

    def reduce_options(self, index: int, values: Set[int]) -> None:
        self.options = list(filter(lambda o: o[index] in values, self.options))


class Skyscrappers:
    def __init__(self, size: int, puzzle_data: List[List[Cell]], rows: List[Line], cols: List[Line]) -> None:
        self.size = size
        self.puzzle_data = puzzle_data
        self.rows = rows
        self.cols = cols
        self.undetermined_count = sum(sum(map(lambda c: c.number is None, line)) for line in self.puzzle_data)

    @staticmethod
    def from_website(website_interface: WebsiteInterface):
        assert website_interface.width == website_interface.height
        size = website_interface.width
        task = website_interface.task
        puzzle_data = [[Cell(size) for i in range(size)] for j in range(size)]
        if "," in task:
            board_numbers = task[task.find(",") + 1:]
            task = task[:task.find(",")]
            cur_board_index = 0
            cur_task_index = 0
            while cur_task_index < len(board_numbers):
                col = cur_board_index % size
                row = cur_board_index // size
                if board_numbers[cur_task_index].isdigit():
                    puzzle_data[row][col] = Cell(size, number=int(board_numbers[cur_task_index]))
                    cur_board_index += 1
                elif "a" <= board_numbers[cur_task_index] <= "z":
                    cur_board_index += ord(board_numbers[cur_task_index]) - 96
                cur_task_index += 1
            assert cur_board_index == size ** 2
        side_numbers = [int(n) if n else None for n in task.split("/")]
        cols = []
        rows = []
        for i in range(size):
            cols.append(Line(size, side_numbers[i], side_numbers[size + i]))
            rows.append(Line(size, side_numbers[2 * size + i], side_numbers[3 * size + i]))
        return Skyscrappers(size, puzzle_data, rows, cols)

    def __deepcopy__(self, _) -> Skyscrappers:
        return Skyscrappers(self.size, deepcopy(self.puzzle_data), deepcopy(self.rows), deepcopy(self.cols))

    def __str__(self) -> str:
        lines = [[""] + list(map(lambda c: str(c.forward_number or ""), self.cols)) + [""]]
        for i, row in enumerate(self.rows):
            lines.append([str(row.forward_number or "")] + self.puzzle_data[i] + [str(row.reverse_number or "")])
        lines.append([""] + list(map(lambda c: str(c.reverse_number or ""), self.cols)) + [""])
        return "\n".join("\t".join(map(str, line)) for line in lines)

    def serialize_solution(self) -> str:
        return ",".join(",".join(map(str, row)) for row in self.puzzle_data)

    def __getitem__(self, key: Tuple[int, int]) -> Cell:
        row, col = key
        return self.puzzle_data[row][col]

    def __setitem__(self, key: Tuple[int, int], value: int):
        row, col = key
        if not self[key].number:
            self.undetermined_count -= 1
        self[key].set(value)
        for i in range(self.size):
            if i != row:
                self[i, col].discard(value)
            if i != col:
                self[row, i].discard(value)

    def reduce_option_in_line(self, line: Line, line_index: int, is_column: bool):
        possible_values_list = line.get_possible_values_list()
        for j, values in enumerate(possible_values_list):
            position = (line_index, j)
            if is_column:
                position = reversed(position)
            self[position].reduce_options(values)
        uncertain_indices = [j for j in range(self.size) if len(possible_values_list) > 1]
        for subset in powerset(uncertain_indices, 2):
            all_possible_values = set.union(*(possible_values_list[j] for j in subset))
            if len(all_possible_values) < len(subset):
                raise SkyscrappersException()
            elif len(all_possible_values) == len(subset):
                for j in set(range(self.size)) - subset:
                    position = (line_index, j)
                    if is_column:
                        position = reversed(position)
                    self[position].discard_set(all_possible_values)

    def get_total_options(self) -> int:
        return sum(sum(len(c.options) for c in row) for row in self.puzzle_data)

    def find_best_place_to_guess(self):
        best_pos = (0, 0)
        best_options = 1000
        best_max = 0
        for i in range(self.size):
            for j in range(self.size):
                cur = self[i, j]
                if 1 < len(cur.options) < best_options or \
                        (len(cur.options) == best_options and max(cur.options) > best_max):
                    best_pos = (i, j)
                    best_options = len(cur.options)
                    best_max = max(cur.options)
        return best_pos

    def guess(self):
        guess_pos = self.find_best_place_to_guess()
        guess = max(self[guess_pos].options)
        try:
            new_board = deepcopy(self)
            new_board[guess_pos].set(guess)
            new_board.solve()
            self.puzzle_data = new_board.puzzle_data
            self.undetermined_count = new_board.undetermined_count
        except SkyscrappersException:
            self[guess_pos].discard(guess)

    def solve(self) -> None:
        last_total_options = self.get_total_options()
        while self.undetermined_count:
            for i, (row, col) in enumerate(zip(self.rows, self.cols)):
                self.reduce_option_in_line(row, i, False)
                self.reduce_option_in_line(col, i, True)

            while True:
                change = False
                for i in range(self.size):
                    for j in range(self.size):
                        cell = self[i, j]
                        self.rows[i].reduce_options(j, cell.options)
                        self.cols[j].reduce_options(i, cell.options)
                        if not cell.number and len(cell.options) == 1:
                            self[i, j] = next(iter(cell.options))
                            change = True
                if not change:
                    break

            total_options = self.get_total_options()
            if total_options == last_total_options and self.undetermined_count:
                self.guess()
            last_total_options = total_options


def main():
    website_interface = WebsiteInterface("https://www.puzzle-skyscrapers.com/?size=9")
    board = Skyscrappers.from_website(website_interface)
    board.solve()
    website_interface.submit_solution(board.serialize_solution())
    print(board)


if __name__ == "__main__":
    main()
