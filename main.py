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
    
    def __str__(self):
        """String representation for print."""
        _str = ""
        m = int(self.n / 3)
        for i in range(self.n):
            if not i % m:
                _str += "#############\n"
            for j in range(self.n):
                if not j % m:
                    _str += "#"
                _str += "{:d}".format(self.grid[i, j]["val"])
            _str += "#\n"
        return _str + "#############"
    
    def copy(self, _grid):
        self.grid = np.array(_grid.grid)
        self.n = int(_grid.n)
    
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
        numbers = [0, 0, 5, 4, 0, 0, 0, 0, 3,
                   0, 0, 0, 2, 1, 5, 0, 0, 0,
                   0, 0, 2, 0, 0, 0, 0, 0, 0,
                   0, 0, 6, 3, 4, 0, 0, 1, 0,
                   0, 3, 4, 0, 0, 0, 2, 8, 0,
                   0, 2, 0, 0, 5, 7, 4, 0, 0,
                   0, 0, 0, 0, 0, 0, 9, 0, 0,
                   0, 0, 0, 1, 3, 8, 0, 0, 0,
                   7, 0, 0, 0, 0, 4, 3, 0, 0]
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
        mSqr = self.grid[(i // m)*m:(i // m + 1)*m,
                         (j // m)*m:(j // m + 1)*m]["options"]
        mRow = self.grid[i, :]["options"]
        mCol = self.grid[:, j]["options"]
        
        # Update those cases
        def doUpdate(_list):
            _list[..., val-1] = 0
        list(map(doUpdate, [mRow, mCol, mSqr]))
        
    
    def getChanges(self, n_target=1):
        """Get coordinates and val of proposed changes to the grid."""
        n_options = np.sum(self.grid["options"], axis=-1)
        x, y = np.where(n_options == n_target)
        _, z = np.where(self.grid[x, y]["options"])
        if n_target > 1:
            z = z.reshape((len(x), n_target))
        return x, y, z + 1

    def solve(self, verbose=False):
        """Solve the sudoku grid until simple wave collapse limit."""
        # 1. Initialise: collapse wave functions around known values
        # TODO: don't repeat this step in later solve() calls
        for i, row in enumerate(self.grid):
            for j, case in enumerate(row):
                val = case.val
                if val:
                    self.propagate(i, j, val)

        # 2. Create list of "known" cases with option length one      
        x, y, z = self.getChanges()
        
        # 3. Propagate and repeat until no viable options are left
        while len(x):
            # Only first because of possible updates
            i, j, val = list(zip(x, y, z))[0]
            if verbose:
                print("Inserting {:d} in [{:d},{:d}]..".format(val, i, j))
            self.propagate(i, j, val)
            x, y, z = self.getChanges()
    
    def isSolved(self):
        """Check if grid is solved."""
        return not 0 in self.grid["val"]
    
    def isValid(self):
        """Check if the grid makes sense."""
        def checkRow(row):
            _, counts = np.unique(row[row != 0], return_counts=True)
            return not np.sum(np.array(counts != 1, dtype=int))

        for i, row in enumerate(self.grid["val"]):
            # Check row
            if not checkRow(row):
                return False
            
            # Check column
            col = self.grid[:, i]["val"]
            if not checkRow(col):
                return False
            
            # Check square
            j, k = [[0, 0], [0, 3], [0, 6],
                    [3, 0], [3, 3], [3, 6],
                    [6, 0], [6, 3], [6, 6]][i]
            m = int(self.n / 3)
            sqr = self.grid[(j // m)*m:(j // m + 1)*m,
                            (k // m)*m:(k // m + 1)*m]["val"].flatten()
            if not checkRow(sqr):
                return False
            
            # Check cases
            for j, val in enumerate(row):
                if not val + np.sum(self.grid[i, j]["options"]):
                    return False
        return True



def branchOut(grid, i, j, values):
    """Branch out between the values and return newly-solved grids."""
    _ret = []
    for val in values:
        _grid = Grid(grid.n)
        _grid.copy(grid)
        _grid.propagate(i, j, val)
        _grid.solve()
        _ret += [_grid]
    return _ret


def createUniverses(grid):
    # Branch out parallel universes
    x, y, z = grid.getChanges(2)
    
    for i, j, values in zip(x, y, z):
        gridOne, gridTwo = branchOut(grid, i, j, values)
        
        # Pick best option, insert, repeat
        g1_solved, g1_valid = gridOne.isSolved(), gridOne.isValid()
        g2_solved, g2_valid = gridTwo.isSolved(), gridTwo.isValid()
    
        if g1_solved and g1_valid:
            return gridOne
        elif g2_solved and g2_valid:
            return gridTwo
        elif g1_valid and not g2_valid:
            return gridOne
        elif g2_valid and not g1_valid:
            return gridTwo
        else:
            continue
    
    print("Best solution:")
    print(grid)
    raise Exception("Sudoku could not be solved!")


def main(nGrid=9):
    g = Grid(nGrid)
    g.loadStd()
    print("Initial grid:")
    print(g, "\n----------------------------")

    # Try to solve directly
    g.solve()
    if g.isSolved(): return
    
    # Solve with branching
    while not g.isSolved():
        newGrid = createUniverses(g)
        g.copy(newGrid)
    print("Solution:\n", g)


if __name__ == '__main__':
    main()
