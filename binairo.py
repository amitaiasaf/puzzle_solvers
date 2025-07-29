from __future__ import annotations
import copy
from enum import Enum
import itertools
import time
from typing import Dict, List, Tuple

from utils.website_interface import WebsiteInterface, WrongSolutionException


class BinarioError(Exception):
    pass



class Cell(Enum):
    UNDETERMINED = ","
    BLACK = "B"
    WHITE = "W"

    def __str__(self) -> str:
        return self.value
    
    def __invert__(self) -> Cell:
        if self == Cell.BLACK:
            return Cell.WHITE
        elif self == Cell.WHITE:
            return Cell.BLACK
        else:
            return Cell.UNDETERMINED


class Constraint(Enum):
    EQUAL = 1
    NOT_EQUAL = 3

class Line(list[Cell]):
    def __init__(self, size: int) -> None:
        super().__init__([Cell.UNDETERMINED for _ in range(size)])
        self.remaining = {Cell.BLACK: size // 2, Cell.WHITE: size // 2}
        self.undetermined = set(range(size))
        self.constraints: Dict[int, Constraint] = {}

    def __str__(self) -> str:
        return " ".join(str(cell) for cell in self)
    

    def set(self, index: int, value: Cell):
        if self[index] == value:
            return
        if self[index] != Cell.UNDETERMINED:
            raise BinarioError()
        self[index] = value
        for i in range(max(0, index - 2), min(len(self) - 2, index + 1)):
            if self[i] == self[i + 1] == self[i + 2]:
                raise BinarioError("Cannot set value that creates a triplet")
        self.remaining[value] -= 1
        if self.remaining[value] < 0:
            raise BinarioError(f"Too many {value} in line")
        if index in self.constraints:
            if self.constraints[index] == Constraint.EQUAL and self[index] == ~ self[index + 1]:
                raise BinarioError("Cannot set value that violates an equal constraint")
            if self.constraints[index] == Constraint.NOT_EQUAL and self[index] == self[index + 1]:
                raise BinarioError("Cannot set value that violates a not-equal constraint")
        
        if index -1 in self.constraints:
            if self.constraints[index - 1] == Constraint.EQUAL and self[index - 1] == ~ self[index]:
                raise BinarioError("Cannot set value that violates an equal constraint")
            if self.constraints[index - 1] == Constraint.NOT_EQUAL and self[index - 1] == self[index]:
                raise BinarioError("Cannot set value that violates a not-equal constraint")

        self.undetermined.remove(index)

    def get_new_certain_values(self) -> Dict[int, Cell]:
        new_values: Dict[int, Cell] = {}
        for color in self.remaining:
            if self.remaining[color] == 0:
                for i in self.undetermined:
                    new_values[i] = ~color
            elif self.remaining[color] == 1:
                for i in range(len(self) - 2):
                    if color not in self[i:i + 3]:
                        for val in self.undetermined - {i, i + 1, i + 2}:
                            new_values[val] = ~color

        for i, c in enumerate(self):
            if i > 0 and i < len(self) - 1:
                if c == Cell.UNDETERMINED and self[i - 1] == self[i + 1] and self[i - 1] != Cell.UNDETERMINED:
                    new_values[i] = ~self[i - 1]
            if i < len(self) - 2:
                if c == Cell.UNDETERMINED and self[i + 1] == self[i + 2] and self[i + 1] != Cell.UNDETERMINED:
                    new_values[i] = ~self[i + 1]
            if i > 1:
                if c == Cell.UNDETERMINED and self[i - 1] == self[i - 2] and self[i - 1] != Cell.UNDETERMINED:
                    new_values[i] = ~self[i - 1]
            
        return new_values
    
    def get_constraints_new_certain_values(self) -> Dict[int, Cell]:
        new_values: Dict[int, Cell] = {}
        for i, constraint in self.constraints.items():
            if constraint == Constraint.EQUAL:
                if self[i] ==Cell.UNDETERMINED and self[i + 1] != Cell.UNDETERMINED:
                    new_values[i] = self[i + 1]
                elif self[i + 1] == Cell.UNDETERMINED and self[i] != Cell.UNDETERMINED:
                    new_values[i + 1] = self[i]
                elif i > 0 and self[i-1] != Cell.UNDETERMINED and self[i] == Cell.UNDETERMINED:
                    new_values[i] = new_values[i + 1] = ~self[i - 1]
                elif i < len(self) - 2 and self[i + 2] != Cell.UNDETERMINED and self[i + 1] == Cell.UNDETERMINED:
                    new_values[i + 1] = new_values[i] = ~self[i + 2] 
            elif constraint == Constraint.NOT_EQUAL:
                if self[i] == Cell.UNDETERMINED and self[i + 1] != Cell.UNDETERMINED:
                    new_values[i] = ~self[i + 1]
                elif self[i + 1] == Cell.UNDETERMINED and self[i] != Cell.UNDETERMINED:
                    new_values[i + 1] = ~self[i]
        return new_values
    
    def __le__(self, value: List[Cell]) -> bool:
        for c1, c2 in zip(self, value):
            if c1 != c2 and c1 != Cell.UNDETERMINED:
                return False
        return True



class Binario:
    def __init__(self, rows: List[Line], cols: List[Line], is_plus: bool = False):
        self.rows = rows
        self.cols = cols
        self.is_plus = is_plus
        self.undetermined_count: int = sum(row.count(Cell.UNDETERMINED) for row in rows)
        for i in range(len(rows)):
            for j in range(len(rows[0])):
                self.cols[j].set(i, rows[i][j])
                

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self.rows)

    @staticmethod
    def create_binario_from_website(website_interface: WebsiteInterface) -> Binario:
        width = website_interface.width
        height = website_interface.height
        rows = [Line(width) for _ in range(height)]
        cols = [Line(height) for _ in range(width)]
        cur_index = 0
        preset, is_plus, constraints = website_interface.task.partition("|")
        for c in preset:
            if c.isdigit():
                c = int(c)
                if c %2 == 0:
                    rows[cur_index // width].set(cur_index % width, Cell.WHITE)
                    cols[cur_index % width].set(cur_index // width, Cell.WHITE)
                    cur_index += 1 
                else:
                    rows[cur_index // width].set(cur_index % width, Cell.BLACK)
                    cols[cur_index % width].set(cur_index // width, Cell.BLACK)
                    cur_index += 1
            else:
                cur_index += ord(c) - ord('a') + 1
        if cur_index != width * height:
            raise BinarioError("Invalid task length")
        if is_plus:
            cur_index = 0
            for c in constraints:
                if c.isdigit():
                    c = int(c)
                    if c >= 6:
                        cols[cur_index % width].constraints[cur_index // width] = Constraint.NOT_EQUAL
                        c -= 6
                    if c >= 3:
                        cols[cur_index % width].constraints[cur_index // width] = Constraint.EQUAL
                        c -= 3
                    if c >= 2:
                        rows[cur_index // width].constraints[cur_index % width] = Constraint.NOT_EQUAL
                        c -= 2
                    if c:
                        rows[cur_index // width].constraints[cur_index % width] = Constraint.EQUAL
                    cur_index += 1
                else:
                    cur_index += ord(c) - ord('a') + 1
            if cur_index != width * height:
                raise BinarioError("Invalid constraints length")
        return Binario(rows, cols, bool(is_plus))

    def serialize_solution(self) -> str:
        return "".join("".join("1" if c == Cell.BLACK else "0" for c in row) for row in self.rows)

    def __setitem__(self, key: Tuple[int, int], value: Cell):
        row, col = key
        self.rows[row].set(col, value)
        self.cols[col].set(row, value)
        self.undetermined_count -= 1

    def avoid_duplicates(self, lines: list[Line], transpose: bool = False):
        completed_lines = list(filter(lambda i: not lines[i].undetermined, range(len(lines))))
        lines_to_check = filter(lambda i: len(lines[i].undetermined) == 2, range(len(lines)))
        for i in lines_to_check:
            for completed_line in completed_lines:
                if lines[i] <= lines[completed_line]:
                    for j, c in enumerate(lines[i]):
                        if c == Cell.UNDETERMINED:
                            self[(i, j) if not transpose else (j, i)] = ~lines[completed_line][j]

        for i, completed_line in enumerate(completed_lines):
            for other_line in completed_lines[i + 1:]:
                if lines[completed_line] == lines[other_line]:
                    raise BinarioError("Duplicate completed lines found")

    def get_guess_score(self, guess: tuple[int, int, Cell]) -> int:
        row, col, value = guess
        if self.rows[row][col] != Cell.UNDETERMINED:
            return 10000
        return self.rows[row].remaining[value] + self.cols[col].remaining[value] \
              - self.rows[row].remaining[~value] - self.cols[col].remaining[~value]
        

    def guess(self):
        new_board = copy.deepcopy(self)
        row, col, value = min(itertools.product(range(len(self.rows)), range(len(self.cols)), [Cell.BLACK, Cell.WHITE]),
                           key = self.get_guess_score)
        if self.rows[row][col] != Cell.UNDETERMINED:
            raise BinarioError("Cannot guess a cell that is already determined")
        
        try:
            new_board[row, col] = value
            new_board.solve()
            self.rows = new_board.rows
            self.cols = new_board.cols
            self.undetermined_count = new_board.undetermined_count
        except BinarioError:
            self[row, col] = ~value


    def solve(self):
        last_undetermined = 0
        while self.undetermined_count:
            if last_undetermined == self.undetermined_count:
                self.guess()
                if self.undetermined_count == 0:
                    return
            last_undetermined = self.undetermined_count

            for i, row in enumerate(self.rows):
                for j, value in row.get_new_certain_values().items():
                    self[i, j] = value

            for j, col in enumerate(self.cols):
                for i, value in col.get_new_certain_values().items():
                    self[i, j] = value
            if not self.is_plus:
                self.avoid_duplicates(self.rows)
                self.avoid_duplicates(self.cols, transpose=True)
            else:
                for i, row in enumerate(self.rows):
                    for j, value in row.get_constraints_new_certain_values().items():
                        self[i, j] = value
                for j, col in enumerate(self.cols):
                    for i, value in col.get_constraints_new_certain_values().items():
                        self[i, j] = value
        if not self.is_plus:
            self.avoid_duplicates(self.rows)
            self.avoid_duplicates(self.cols, transpose=True)


def main():
    for i in range(1000):
        website_interface = WebsiteInterface("https://www.puzzle-binairo.com/binairo-plus-20x20-hard/",
                                             params={"specific": "0", "specid": "334,131"})
        board = Binario.create_binario_from_website(website_interface)
        try:
            start = time.time()
            board.solve()
            print(f"Successfully solved Binario puzzle (puzzle id {website_interface.puzzle_id}) in {time.time() - start:.06f} seconds: \n{board}")
        except BinarioError as e:
            print(f"Error solving Binario puzzle (puzzle id {website_interface.puzzle_id}): \n{board}")
            print(e)
            return


        try:
            website_interface.submit_solution(board.serialize_solution())
            print(f"Submitted solution for Binario puzzle (puzzle id {website_interface.puzzle_id})")
        except WrongSolutionException as e:
            print("The solution is incorrect, please check the logic.")
            print("Task data:", website_interface.task)
            


if __name__ == "__main__":
    main()
