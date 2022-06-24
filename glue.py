"""Create a bridge between image processing and number detection."""
import numpy as np
import torch
from imgParser_v2 import getSquares
from network import imageClassifier
from matplotlib import pyplot as plt
from skimage import transform
from grid import Grid


def main(image_path="images/testSudoku.png",
         network_path="results/model.pth"):
    """Get images from imgParser.py and evaluate using network.py model."""
    squares = getSquares(image_path)
    network = imageClassifier()
    network.load_state_dict(torch.load(network_path))
    
    # First test
    # img = transform.resize(squares[0][2], (28, 28))
    # img = 255 - img
    # mu, sigma = np.mean(img), np.std(img)
    # img = (img - mu) / sigma
    # img[img < .5] = 0
    # plt.imshow(img)
    
    # network.eval()
    # with torch.no_grad():
    #     data = torch.from_numpy(img[None, None, ...]).float()        
    #     output = network(data)
    #     _, prediction = output.data.max(1, keepdim=True)
    
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
            
            if (i == 0 and j == 3) or (i == 8 and j == 2):
                plt.figure()
                plt.imshow(img)
                data = np.array(output.data[0])
                print(100 / data / np.sum(1 / data))
    
    testGrid = Grid()
    testGrid.loadStd(numbers_for_sudoku)
    print(testGrid)        


if __name__ == '__main__':
    main()
