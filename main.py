"""Solving sudokus."""
import numpy as np
from grid import Grid


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


def createUniverses(grid, n_options=2):
    """For each multi-option case, try each option in parallel."""
    # Get changes for n=2, if stuck then continue to n=3 etc..
    # Continue until one change is implemented, then return to let n=2 work
    
    x = [-1]
    while len(x) > 0:
        x, y, z = grid.getChanges(n_options)
        
        for i, j, values in zip(x, y, z):
            grids = branchOut(grid, i, j, values)
            
            grids_solved = np.array(list(map(lambda x: x.isSolved(), grids)),
                                    dtype=int)
            grids_valid = np.array(list(map(lambda x: x.isValid(), grids)),
                                   dtype=int)
            
            # If one is solved and valid, return
            combination = grids_solved + grids_valid
            where_combination = np.where(combination == 2)
            if len(where_combination[0]) > 0:
                return grids[where_combination[0][0]]
    
            # If only one is valid, return
            if np.sum(grids_valid) == 1:
                return grids[np.where(grids_valid)[0][0]]
        
        n_options += 1
        

def main(nGrid=9):
    """Load a sudoku grid then solve it."""
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
    print("Solution:")
    print(g)


if __name__ == '__main__':
    main()
