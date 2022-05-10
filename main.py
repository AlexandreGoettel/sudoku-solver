"""Solving sudokus."""
import numpy as np


class grid:
    """Hold the sudoku grid and helpful functions."""

    def __init__(self):
        """Initialise the grid."""
        dcase = np.dtype([('val', int), ('options', list), ('n', int)])
        self.grid = np.zeros((9, 9), dtype=dcase)
        print(self.grid)

    def getRow(self, i):
        """Return a row of the Sudoku grid."""
        return self.grid[i, :]

    def getColumns(self, i):
        """Return a column of the Sudoku grid."""
        return self.grid[:, i]


class case:
    """Hold grid value and "wave function"."""

    def __init__(self, val=0):
        """Initialise case with val and all options."""
        self.val = val
        self.options = np.arange(9) + 1
        self.n = len(self.options)

    def updateOption(self, new_options):
        """Update value of self.options and accompanying self.n."""
        self.options = new_options
        self.n = len(self.options)

    def removeOption(self, val):
        """
        Remove val from options.

        val: int or array of int or np.array of int
        """
        if isinstance(val, (type([]), type(np.array([], dtype=int)))):
            for n in val:
                self.removeOption(n)
        elif isinstance(val, int):
            if val in self.options:
                self.updateOption(np.delete(self.options,
                                            np.where(self.options == val)))
        else:
            raise TypeError("val should be int or array of int.")


def main():
    g = grid()
    print(g.grid[0, 0].val)


if __name__ == '__main__':
    main()
