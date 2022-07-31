"""Get font images with labels from different samples, prep for network."""
import os
import utils
import numpy as np
from skimage import io


def getFontNumbers(fontdir="fonts"):
    """From all the images in fontdir, get separated images of the ints."""
    font_images = os.listdir(fontdir)

    out_numbers = []
    for font in font_images:
        # Read-in image
        im = io.imread(os.path.join(fontdir, font))
        im = utils.rgb2gray(im)
    
        # Crop left/right on projection hard cut
        proj = np.sum(im < 100, axis=0) == 0
        limits = []
        for i, val in enumerate(proj[1:-2]):
            if proj[i] != val and val == proj[i+2]:
                if val:
                    limits += [i]
                else:
                    limits += [i-1]

        # Crop top/bottom
        proj = list(np.sum(im < 100, axis=1) == 0)
        try:
            upper = proj.index(0)
        except ValueError:
            upper = 0
        try:
            lower = proj[upper:].index(1) + upper - 1
        except ValueError:
            lower = len(proj) - 1

        if len(limits) == 19:
            limits += [im.shape[1]-1]
        assert len(limits) == 20
        numbers = []
        for i in range(9):
            numbers += [im[upper:lower+1, limits[2*i]:limits[2*i+1]]]
        out_numbers += [numbers]

    return out_numbers, [[1, 2, 3, 4, 5, 6, 7, 8, 9]*len(out_numbers)]
