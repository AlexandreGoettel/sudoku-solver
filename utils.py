"""Collection of useful functions needed by different modules."""
import numpy as np


def rgb2gray(rgb):
    """Convert color image to grayscale."""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def getIntersect(line1, line2):
    """Get the intersect of two lines, each defined as a 2D pos and slope."""
    (x0, y0, s0), (x1, y1, s1) = line1, line2
    x = (s0*x0 - s1*x1 + y1 - y0) / (s0 - s1)
    y = (s0*s1*(x0 - x1) + s0*y1 - s1*y0) / (s0 - s1)
    return x, y
