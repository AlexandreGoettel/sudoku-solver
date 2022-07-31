"""New imgParser using a deterministic approach with Hough-transform."""
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, filters, transform
from utils import rgb2gray, getIntersect


def getBounds(_im, axis=0, _stdlim=5, _xlim=0.95):
    """Get bounds of innermost square on axis."""
    # Get derivative of projected sum
    x = np.sum(_im, axis=axis)
    dx = x[1:] - x[:-1]
    
    # Filter out inner noise
    dx_filter = dx[np.abs(dx - np.mean(dx)) < np.std(dx, ddof=1)]

    # Go from the outside to exit of first turbulent region
    n, N, _xmax = int(len(dx) / 2), len(x), _xlim*max(x)
    _stdmax = _stdlim * np.std(dx_filter, ddof=1)
    _in_region = False
    for i, val in enumerate(dx[:n]):
        if np.abs(val) > _stdmax:
            _in_region = True
        elif _in_region and x[i-1] >= _xmax:
            _left = i + 1
            break
    
    _in_region = False
    for i, val in enumerate(dx[n:][::-1]):
        if np.abs(val) > _stdmax:
            _in_region = True
        elif _in_region and x[N-i-1] >= _xmax:
            _rght = len(x) - i
            break
    
    return _left, _rght


def getSquares(path, verbose=False):
    """
    Detect and extract squares using Hough transform.
    
    Goal of this function is just to return the squares with numbers
    (and their respective coordinates)
    Further transformations/normalisations etc. are handled by network.py
    """
    print("Reading the image..")
    im = io.imread(path)
    
    print("Pre-processing..")
    grayscale = rgb2gray(im)
    # grayscale = filters.gaussian(grayscale, sigma=2)
    im = grayscale > 100
    
    print("Begin line finding..")
    edges = ~im
    angles = np.append(np.linspace(np.pi/2*0.9, np.pi/2*1.1, 150),
                        np.linspace(-np.pi/2*0.1, np.pi/2*0.1, 150))
    h, theta, d = transform.hough_line(edges, theta=angles)
    
    x0s, y0s, slopes = [], [], []
    if verbose:
        plt.imshow(im, cmap="gray", vmin=0, vmax=1)
    for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d)):
        (x0, y0) = dist *  np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi/2)
        
        x0s += [x0]
        y0s += [y0]
        slopes += [slope]
    
    # Convert to convenient numpy format    
    data = np.column_stack([x0s, y0s, slopes])
    print("Number of lines found: {}/20".format(data.shape[0]))
    
    # Clean results
    if data.shape[0] < 20:
        # TODO: Can infer missing lines from rest in theory
        raise TypeError("Something went wrong: not enough lines detected..")
    
    while data.shape[0] > 20:
        # First separate horizontal and vertical lines
        mask = np.abs(data[:, 2]) < 1
        horizontal, vertical = np.sort(data[mask, 1]), np.sort(data[~mask, 0])
        
        # Remove element with smallest distance to neighbour
        gaps = np.append(np.abs(horizontal[1:] - horizontal[:-1]),
                         np.abs(vertical[1:] - vertical[:-1]))
        iMin = np.argmin(gaps)
        hMin = horizontal[np.argmin(gaps)] if iMin < len(horizontal) else\
            vertical[np.argmin(gaps) - len(horizontal) + 1]
        data = np.delete(data, obj=np.where(data == hMin)[0], axis=0)
        
    # Now isolate the images
    idx = np.argsort(data[:, 1])
    data = data[idx[::-1], :]
    
    # Separate horizontal and vertical lines
    mask = np.abs(data[:, 2]) < 1
    horizontal, vertical = data[mask, :], data[~mask, :]
    
    # Sort
    horizontal = horizontal[np.argsort(horizontal[:, 1]), :]
    vertical = vertical[np.argsort(vertical[:, 0]), :]
    assert(len(horizontal) == 10 and len(vertical) == 10)
    
    squares = []
    print("Extracting squares..")
    x, y = np.arange(im.shape[0]), np.arange(im.shape[1])
    X, Y = np.meshgrid(y, x)
    for i in range(9):
        for j in range(9):
            # Get bounding lines
            horizontal_0, horizontal_1 = horizontal[i, :], horizontal[i+1, :]
            vertical_0, vertical_1 = vertical[j, :], vertical[j+1, :]
            
            # Construct bounding box for square
            xa, ya = getIntersect(horizontal_0, vertical_0)
            xb, yb = getIntersect(horizontal_0, vertical_1)
            xc, yc = getIntersect(horizontal_1, vertical_1)
            xd, yd = getIntersect(horizontal_1, vertical_0)
            _up, _lo = int(max(yc, yd)), int(min(ya, yb))
            _left, _rght = int(min(xa, xd)), int(max(xb, xc))
            _square = im[_lo:_up+1, _left:_rght+1]

            # Now remove exterior borders
            # Vertical lines to y axis
            theta = -np.arctan((vertical_0[-1] + vertical_1[-1]) / 2.)
            _square = transform.rotate(_square, angle=theta)
            left, rght = getBounds(_square, axis=0)
            
            # Horizontal lines to x axis
            phi = np.arctan((horizontal_0[-1]+horizontal_1[-1])/2)
            _square = transform.rotate(_square, angle=phi - theta)     
            lo, up = getBounds(_square, axis=1)

            square = _square[lo:up+1, left:rght+1]
            
            # Check if there is a number in there
            res = transform.resize(square, (28, 28)) > 0.5
            if not 28*28 - np.sum(res):
                continue
            
            # TODO: return part from orig (grayscale) img
            squares += [(i, j, grayscale[_lo:_up+1, _left:_rght+1]
                         [lo:up+1, left:rght+1])]
    
    return squares


if __name__ == '__main__':
    squares = getSquares("images/testSudoku.png")
