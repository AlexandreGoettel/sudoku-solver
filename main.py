"""Solving sudokus."""
import numpy as np


class Grid:
    """Hold the sudoku grid and helpful functions."""

    def __init__(self, n=9):
        """Init an empty grid."""
        dt = np.dtype([('val', int), ('options', int, (n,))])
        a = np.zeros(shape=(n, n), dtype=dt)
        self.grid = a.view(np.recarray)
        self.n = n
        self.resetOptions()
    
    def resetOptions(self):
        """"Set options to one everywhere."""
        for i in range(self.n):
            for j in range(self.n):
                self.grid[i, j]["options"] = np.ones(self.n)

    def readFromCmdl(self):
        """Read a sudoku grid from command line."""
        for i in range(self.n):
            for j in range(self.n):
                # TODO: error handling
                val = int(input("Enter the next number: "))
                if val:
                    self.setVal(val, i, j)
                else:
                    # Maybe move this to init?
                    self.grid[i, j]["options"] = np.ones(self.n)
    
    def loadStd(self):
        """Load a standard sudoku grid for debugging."""
        numbers = [0, 0, 0, 0, 4, 0, 9, 0, 1,
                   0, 0, 0, 0, 0, 9, 8, 0, 6,
                   0, 0, 0, 0, 3, 0, 0, 7, 0,
                   5, 0, 0, 0, 0, 3, 0, 0, 8,
                   0, 1, 8, 9, 0, 5, 4, 2, 0,
                   9, 0, 0, 1, 0, 0, 0, 0, 7,
                   0, 3, 0, 0, 8, 0, 0, 0, 0,
                   1, 0, 2, 7, 0, 0, 0, 0, 0,
                   7, 0, 9, 0, 5, 0, 0, 0, 0]
        for m, val in enumerate(numbers):
            i, j = m // self.n, m % self.n
            self.setVal(val, i, j)
    
    def setVal(self, val, i, j):
        """Set a value at a particular grid coord."""
        case = self.grid[i, j]
        case["val"], case["options"] = val, np.zeros(self.n)
    
    def solve(self):
        # TODO


def main():
    g = Grid(9)
    g.loadStd()
    g.solve()


if __name__ == '__main__':
    main()
