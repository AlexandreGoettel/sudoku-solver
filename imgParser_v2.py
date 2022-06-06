"""New imgParser using a deterministic approach with Hough-transform."""
from matplotlib import pyplot as plt
import numpy as np
import scipy
from skimage import io, filters, feature, transform
import time


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def test_proba():
    """Test the probabilistic Hough transform."""
    print("Reading the image..")
    im = io.imread("images/testSudoku.png")
    
    print("Pre-processing..")
    im = rgb2gray(im)
    im = filters.gaussian(im, sigma=2)
    im = im > 100
    plt.imshow(im, cmap="gray", vmin=0, vmax=1)

    print("Begin line finding..")
    # edges = feature.canny(im, sigma=0, low_threshold=0, high_threshold=1)
    edges = ~im
    angles = np.append(np.linspace(np.pi/2*0.9, np.pi/2*1.1, 50),
                        np.linspace(-np.pi/2*0.1, np.pi/2*0.1, 50))
    lines = transform.probabilistic_hough_line(
        edges, threshold=1, line_length=2000, line_gap=50, theta=angles)
    
    
    # Plot results
    plt.imshow(~edges, cmap="gray", vmin=0, vmax=1)
    for line in lines:
        p0, p1 = line
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]])
    
    plt.savefig("plots/test_hough.pdf")


def getIntersect(line1, line2):
    """Get the intersect of two lines, each defined as a 2D pos and slope."""
    (x0, y0, s0), (x1, y1, s1) = line1, line2
    x = (s0*x0 - s1*x1 + y1 - y0) / (s0 - s1)
    y = (s0*s1*(x0 - x1) + s0*y1 - s1*y0) / (s0 - s1)
    return x, y


def test_direct():
    """Test the line Hough transform."""
    print("Reading the image..")
    im = io.imread("images/testSudoku.png")
    
    print("Pre-processing..")
    im = rgb2gray(im)
    # im = filters.gaussian(im, sigma=2)
    im = im > 100
    
    print("Begin line finding..")
    edges = ~im
    angles = np.append(np.linspace(np.pi/2*0.9, np.pi/2*1.1, 150),
                        np.linspace(-np.pi/2*0.1, np.pi/2*0.1, 150))
    h, theta, d = transform.hough_line(edges, theta=angles)
    
    x0s, y0s, slopes = [], [], []
    plt.imshow(im, cmap="gray", vmin=0, vmax=1)
    for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d)):
        (x0, y0) = dist *  np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi/2)
        # plt.axline((x0, y0), slope=slope, color="r")
        
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
        
    # Now isolate the images!
    idx = np.argsort(data[:, 1])
    data = data[idx[::-1], :]
    
    # Separate horizontal and vertical
    mask = np.abs(data[:, 2]) < 1
    horizontal, vertical = data[mask, :], data[~mask, :]
    
    # Sort
    horizontal = horizontal[np.argsort(horizontal[:, 1]), :]
    vertical = vertical[np.argsort(vertical[:, 0]), :]
    
    # plt.figure()
    # plt.scatter(horizontal[:, 0], horizontal[:, 1])
    # plt.scatter(vertical[:, 0], vertical[:, 1])
    assert(len(horizontal) == 10 and len(vertical) == 10)
    
    def plotLine(x, y, s):
        plt.axline((x, y), slope=s, color="r")
        
    for i in range(9):
        for j in range(9):
            horizontal_0, horizontal_1 = horizontal[i, :], horizontal[i+1, :]
            vertical_0, vertical_1 = vertical[j, :], vertical[j+1, :]
            
            xa, ya = getIntersect(horizontal_0, vertical_0)
            xb, yb = getIntersect(horizontal_0, vertical_1)
            xc, yc = getIntersect(horizontal_1, vertical_1)
            xd, yd = getIntersect(horizontal_1, vertical_0)
            plt.scatter([xa, xb, xc, xd], [ya, yb, yc, yd])
            # plt.scatter([horizontal_0[0], horizontal_1[0], vertical_0[0], vertical_1[0]],
            #             [horizontal_0[1], horizontal_1[1], vertical_0[1], vertical_1[1]])
            # for line in horizontal_0, horizontal_1, vertical_0, vertical_1:
            #     plotLine(*line)
            # plt.imshow(im, cmap="gray", vmin=0, vmax=1)
            # plt.draw()
            # plt.pause(.01)
            
            # Extract square
            # t0 = time.time()
            # mask = np.zeros_like(im)
            # [x0, y0, s0], [x1, y1, s1] = horizontal_0, horizontal_1
            # for j in range(mask.shape[1]):
            #     _up = s0*(j - x0) + y0
            #     _lo = s1*(j - x1) + y1
            #     mask[:int(_up), j] = 1
            #     mask[int(_lo):, j] = 1
            
            # im[mask] = 0
            # t1 = time.time()
            # plt.imshow(im, cmap="gray", vmin=0, vmax=1)
            
            t2 = time.time()
            x, y = np.arange(im.shape[0]), np.arange(im.shape[1])
            X, Y = np.meshgrid(y, x)
            
            [x0, y0, s0], [x1, y1, s1] = horizontal_0, horizontal_1
            _up, _lo = s0*(y - x0) + y0, s1*(y - x1) + y1
            m1 = (Y <= _up) ^ (Y >= _lo)
            
            [x0, y0, s0], [x1, y1, s1] = vertical_0, vertical_1
            _left, _rght = (x - y0)/s0 + x0, (x - y1)/s1 + x1
            m2 = (X.T <= _left) ^ (X.T >= _rght)

            mask = (m1 == 1) | (m2.T == 1)
            # im[mask] = 0
            # plt.imshow(im, cmap="gray", vmin=0, vmax=1)
            # return
            
            # [x0, y0, s0], [x1, y1, s1] = vertical_0, vertical_1
            # _left, _rght = (y - y0)/s0 + x0, (y - y1)/s1 + x1
            # m1 = np.any((X[None, :] >= _lo) ^ (X[None, :] <= _up), axis=0).T
            # m2 = np.any((Y[:, None].T >= _rght) ^ (Y[:, None].T <= _left), axis=1)
            # mask = (m1 == 1) | (m2 == 1)
            
            _im = np.array(im)
            _im[mask] = 0
            plt.imshow(_im, cmap="gray", vmin=0, vmax=1)
            plt.draw()
            plt.pause(.001)
        


if __name__ == '__main__':
    test_direct()
