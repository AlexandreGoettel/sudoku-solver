"""Collection of useful functions needed by different modules."""
import numpy as np
from scipy.special import betaln


def rgb2gray(rgb):
    """Convert color image to grayscale."""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def getIntersect(line1, line2):
    """Get the intersect of two lines, each defined as a 2D pos and slope."""
    (x0, y0, s0), (x1, y1, s1) = line1, line2
    x = (s0*x0 - s1*x1 + y1 - y0) / (s0 - s1)
    y = (s0*s1*(x0 - x1) + s0*y1 - s1*y0) / (s0 - s1)
    return x, y


def getEff(m, n):
    """Calculate a HEP-style efficiency with uncertainty."""
    eff = m / float(n)
    var = np.exp(betaln(m+3, n-m+1) - betaln(m+1, n-m+1)) -\
        np.exp(2*(betaln(m+2, n-m+1) - betaln(m+1, n-m+1)))

    return eff, np.sqrt(var)


def getAccuracy(output, labels):
    """Convert using one-hot then eval. accuracy with uncertainty."""
    idx = np.argmax(output, axis=1)
    return getEff(np.sum(np.array(idx == labels, dtype=int)),
                  float(len(labels)))


def weighedAverage(x, sigma):
    """Return average of x weighed with sigma, with uncertainty."""
    numerator = np.sum(x / sigma**2)
    denominator = np.sum(1. / sigma**2)
    return numerator / denominator, 1. / np.sqrt(denominator)


def getOutSize(in_size, kernel_conv, kernel_maxp, n_layers, raw=False):
    for n in range(n_layers):
        in_size = (in_size - (kernel_conv - 1)) / kernel_maxp
        if not raw and in_size % 2:
            in_size += 1  # Add padding of 1
    return int(in_size)


def getChi(x, y, sigma, f, popt):
    """Return chi-square/ndof value between y and f(x, *popt)."""
    y_model, N = f(x, *popt), len(y)

    chi = np.sum(((y - y_model)/sigma)**2) / (N - len(popt))
    return chi
