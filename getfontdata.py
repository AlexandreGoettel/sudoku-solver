"""Get font images with labels from different samples, prep for network."""
# Standard imports
import os
import numpy as np
from skimage import io
import torch
import torchvision
# Project
import utils


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
            numbers += [im[upper:lower+1, limits[2*i]:limits[2*i+1]+1]]
        out_numbers += [numbers]

    labels = np.arange(1, 10).reshape((1, 9)).repeat(len(out_numbers), axis=0)
    return out_numbers, labels


def prepareFontNumbers(mnist_data, fontdir="fonts", generate=False): 
    def getUpLow(im, axis):
        proj = list(np.sum(im > 10, axis=axis) == 0)
        try:
            upper = proj.index(0)
        except ValueError:
            upper = 0
        try:
            lower = proj[upper:].index(1) + upper - 1
        except ValueError:
            lower = len(proj) - 1
        return upper, lower

    # Get average position of img borders in broader picture
    if generate:
        dist = np.zeros(4)  # up, lo, left, rght
        for im in mnist_data:        
            upper, lower = getUpLow(im, axis=1)
            left, right = getUpLow(im, axis=0)
            dist += upper, lower, left, right
        dist /= mnist_data.shape[0]
    else:
        dist = [4.8, 23.5, 6.6, 21.1]
    
    # Convert to left-top-right-bottom padding
    x, y = mnist_data.shape[1:]
    padding = np.round([dist[2]-1, dist[0]-1, y-dist[3], x-dist[1]])
    
    # Now apply to font images
    fontNumbers, fontLabels = getFontNumbers(fontdir)
    N = len(fontNumbers)*9
    images, labels = np.zeros((N, 28, 28)), np.zeros(N, dtype=np.int16)
    for i, font in enumerate(fontNumbers):
        for j, number in enumerate(font):
            im = 255 - number
            
            # Scale
            _, b, _, d = padding
            A = im.shape[0] / (x - b - d)
        
            # First pad, then resize
            im = torch.from_numpy(im[None, ...])
            im = torchvision.transforms.functional.pad(
                im, tuple(np.array(np.round(padding*A), dtype=int)))
            im = torchvision.transforms.Resize(x)(im)  # not always (28, 28)
            im = torchvision.transforms.CenterCrop(x)(im)  # keep aspect ratio
            
            n = i*9 + j
            images[n, ...] = im.numpy()[0, ...]
            labels[n] = fontLabels[i, j]
    return images, labels


class fontData(torch.utils.data.Dataset):
    def __init__(self, mnist_reference):
        super(fontData).__init__()
        self.images, self.labels = prepareFontNumbers(mnist_reference)
    
    def __getitem__(self, i):
        return self.images[i], self.labels[i]
    
    def __len__(self):
        return len(self.labels)
