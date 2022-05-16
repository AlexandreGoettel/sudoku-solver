"""Solving sudokus."""
import numpy as np


class Grid:
    """Hold the sudoku grid and helpful functions."""

    def __init__(self, n=9):
        """Init an empty grid."""
        assert not n % 3

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
        case["val"] = val
        case["options"] = np.zeros(self.n) if val else np.ones(self.n)
    
    def propagate(self, i, j, val):
        """Insert a value in a case and update neighbouring options."""
        # Update case
        self.grid[i, j]["val"] = val
        self.grid[i, j]["options"] = np.zeros(self.n)
        
        # Get options of neighbouring cases
        m = int(self.n / 3)
        (i // m)*m
        mSqr = self.grid[(i // m)*m:(i // m + 1)*m,
                         (j // m)*m:(j // m + 1)*m]["options"]
        mRow = self.grid[i, :]["options"]
        mCol = self.grid[:, j]["options"]
        
        # Update those cases
        def doUpdate(_list):
            _list[..., val-1] = 0
        list(map(doUpdate, [mRow, mCol, mSqr]))
        
    
    def solve(self):
        """Solve the sudoku grid."""
        # 1. Initialise
        # Collapse wave functions around known values
        for i, row in enumerate(self.grid):
            for j, case in enumerate(row):
                val = case.val
                if val:
                    self.propagate(i, j, val)
        
        n_options = np.sum(self.grid["options"], axis=-1)
        print(np.where(n_options == 1))
        self.propagate(4, 8, 3)
        n_options = np.sum(self.grid["options"], axis=-1)
        print(np.array(n_options == 1, dtype=int))
        print(np.where(n_options == 1))
        print(self.grid[4, 0])
        self.propagate(4, 0, 6)
        n_options = np.sum(self.grid["options"], axis=-1)
        print(np.array(n_options == 1, dtype=int))
        print(np.where(n_options == 1))
        # 2. Create list of "known" variables
        
        # 3. Continue until no viable options are left
        # 4. ???
        # 5. Profit

def main():
    g = Grid(9)
    g.loadStd()
    print(g.grid["val"], "\n----------------------------")
    g.solve()


if __name__ == '__main__':
    main()
