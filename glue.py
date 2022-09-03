"""Create a bridge between image processing and number detection."""
import numpy as np
import torch
from imgParser import getSquares
from network_v3 import ImageClassifier
from matplotlib import pyplot as plt
from skimage import transform
from grid import Grid


# TODO: read network kwargs from file instead
def main(image_path="images/testSudoku.png",
         network_path="results/model.pth",
         network_kwargs={"kernel_size": 3, "n_layers": 3, "fcn_mid": 250,
                         "channels_per_layer": 10}):
    """Get images from imgParser.py and evaluate using network.py model."""
    squares = getSquares(image_path)
    network = ImageClassifier(**network_kwargs)
    network.load_state_dict(torch.load(network_path))

    # Get numbers from each squares using network
    numbers_for_sudoku = np.zeros(81)
    network.eval()
    with torch.no_grad():
        for i, j, square in squares:
            # Resize and normalise square
            img = transform.resize(square, (28, 28))
            img = 255 - img
            mu, sigma = np.mean(img), np.std(img, ddof=1)
            img = (img - mu) / sigma
            img[img < .5] = 0  # cleanup

            # Get grid number
            data = torch.from_numpy(img[None, None, ...]).float()
            output = network(data)
            _, [[n]] = output.data.max(1, keepdim=True)
            numbers_for_sudoku[i*9+j] = n

    # Insert in Grid object to solve!
    testGrid = Grid()
    testGrid.loadStd(numbers_for_sudoku)
    print(testGrid)


if __name__ == '__main__':
    main(network_path="playing_fonts.pth")
