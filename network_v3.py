"""Train MNIST based on https://nextjournal.com/gkoehler/pytorch-mnist."""
# Standard imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import betaln
# Torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
# Project imports
from  getfontdata import FontData


class ImageClassifier(nn.Module):
    
    def __init__(self, kernel_size=5, dropout=.3, n_layers=2, fcn_mid=50,
                 channels_per_layer=10):
        """
        Generate an image classifier.
        
        kernel_size: for the conv layers
        n_layers: number of conv layers
        fcn_mid: number of neurons in "hidden" fcn
        channels_per_layer: number of channels added in each convolution
        dropout: given as float between 0 and 1
        """
        super(ImageClassifier, self).__init__()
        # Hyper-params
        self.dropout = dropout
        self.n_layers = n_layers
        # Network
        for n in range(n_layers):
            setattr(self, "conv{}".format(n+1), nn.Conv2d(
                max(1, n*channels_per_layer), (n+1)*channels_per_layer,
                kernel_size=kernel_size, stride=1, padding="valid"))

        self.conv_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(320, fcn_mid)
        self.fc2 = nn.Linear(fcn_mid, 10)
    
    def forward(self, x):
        for n in range(self.n_layers):
            layer = getattr(self, "conv{}".format(n+1))(x)
            if n == self.n_layers - 1:  # dropout on last conv layer
                x = F.leaky_relu(F.max_pool2d(self.conv_drop(layer), 2))
            else:
                x = F.leaky_relu(F.max_pool2d(layer, 2))
        x = x.view(-1, 320)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc2(x)
        return F.log_softmax(x, -1)


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


def batch_normalise(data):
    nBatch = data.shape[0]
    numpy_data = np.sum(data.detach().numpy(), axis=0)    
    
    mu = np.mean(numpy_data) / nBatch
    sigma = np.std(numpy_data.flatten(), ddof=1)
    data = (data - mu) / sigma
    return data


def train(train_loader, val_loader, test_loader, model_path="mnist_test1.pth",
          dropout=.3, lr=.001, momentum=.9, lr_factor=.1, epochs=10,
          patience=5, kernel_size=5, nConvLayers=2, fcn_mid=50, nChannels=10):
    """Train the "ImageClassifier" using the passed datasets."""
    model = ImageClassifier(dropout=dropout, kernel_size=kernel_size,
                            n_layers=nConvLayers, fcn_mid=fcn_mid,
                            channels_per_layer=nChannels)
    
    # define optimiser, criterion
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                     factor=lr_factor,
                                                     patience=patience,
                                                     cooldown=0,
                                                     verbose=True)    
    
    # Training loop
    book = np.zeros((epochs, 2, 3))  # (train, val), (acc, sig, loss)
    nTrain, nValid, nTest = list(map(len, [
        train_loader, val_loader, test_loader]))
    # TODO: not same number of batches!
    train_accuracy, train_loss = np.zeros((2, nTrain)), np.zeros(nTrain)
    val_accuracy, val_loss = np.zeros((2, nValid)), np.zeros(nValid)
    test_accuracy, test_loss = np.zeros((2, nTest)), np.zeros(nTest)
    for epoch in range(epochs):
        print("Epoch: {:d}".format(epoch+1))
        # Switch to training mode
        model.train()
        for ibatch, (data, truth) in enumerate(train_loader):
            optimiser.zero_grad()
            data = batch_normalise(data)
            output = model(data)
            loss = criterion(output, truth)
            
            # Update weights
            loss.backward()
            optimiser.step()
            
            # Book-keeping
            output = output.detach().numpy()
            train_accuracy[:, ibatch] = getAccuracy(output, truth.numpy())
            train_loss[ibatch] = loss.item()

        book[epoch, 0, :2] = weighedAverage(*train_accuracy)
        book[epoch, 0, 2] = np.mean(train_loss)
 
        print("[TRAIN] acc: {:.3f} +- {:.3f}\n[TRAIN] loss: {:.4f}".format(
            *book[epoch, 0, :]))

        # Switch to validation mode
        model.eval()
        for ibatch, (data, truth) in enumerate(val_loader):
            data = batch_normalise(data)
            output = model(data)
            loss = criterion(output, truth)
            
            # Book-keeping
            output = output.detach().numpy()
            val_accuracy[:, ibatch] = getAccuracy(output, truth.numpy())
            val_loss[ibatch] = loss.item()        
        book[epoch, 1, :2] = weighedAverage(*val_accuracy)
        book[epoch, 1, 2] = np.mean(val_loss)
        print("[VALID] acc: {:.3f} +- {:.3f}\n[VALID] loss: {:.4f}".format(
            *book[epoch, 1, :]))
        
        # Update scheduler
        scheduler.step(book[epoch, 1, 2])
        
        # Save model progress
        torch.save(model.state_dict(), model_path)
        
        # Update plots
        if epoch < 2:
            continue
        fig = plt.figure()
        gs = GridSpec(2, 3, figure=fig)
        axAcc, axLoss = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
        axLoss.set_yscale("log")
        
        x = np.arange(epoch+1)
        labels = ["Training", "Validation"]
        for i in range(2):
            axAcc.errorbar(x+i*.1, book[:epoch+1, i, 0], book[:epoch+1, i, 1],
                           fmt=".", label=labels[i], color="C{}".format(i))
            axLoss.plot(x, book[:epoch+1, i, 2], "-",
                        label=labels[i], color="C{}".format(i))

        axAcc.legend(loc="upper left")
        axLoss.legend(loc="upper right")
        plt.show()
    
    # Testing phase
    model.eval()
    for ibatch, (data, truth) in enumerate(test_loader):
        data = batch_normalise(data)
        output = model(data)
        loss = criterion(output, truth)
        
        # Book-keeping
        output = output.detach().numpy()
        test_accuracy[:, ibatch] = getAccuracy(output, truth.numpy())
        test_loss[ibatch] = loss.item()
    results = np.zeros(3)
    results[:2] = weighedAverage(*val_accuracy)
    results[2] = np.mean(test_loss)
    print("[TEST] acc: {:.3f} +- {:.3f}\n[TEST] loss: {:.4f}".format(
        *results))


def main():
    """Do the training on mnist."""
    # Set hyperparameters
    batch_size_train = 64
    validation_size = 5000
    
    kwargs_train = dict(
        kernel_size=5, nConvLayers=2, fcn_mid=50, nChannels=10,  # Train
        epochs=500, patience=10, lr_factor=.1, momentum=.9, lr=.001,  # Model
        dropout=.3)  # Model
    

    # Get training, validation, and test data from mnist
    mnist_train_data = torchvision.datasets.MNIST(
        "mnist/", train=True, download=True,
        transform=torchvision.transforms.ToTensor())
    
    mnist_train_data, mnist_val_data = torch.utils.data.random_split(
        mnist_train_data, [len(mnist_train_data) - validation_size,
                           validation_size])

    mnist_train_loader = torch.utils.data.DataLoader(
        mnist_train_data, batch_size=batch_size_train, shuffle=True)
    
    mnist_val_loader = torch.utils.data.DataLoader(
        mnist_val_data, batch_size=batch_size_train, shuffle=True)

    mnist_test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "mnist/", train=False, download=True,
            transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size_train, shuffle=True)
    
    train(mnist_train_loader, mnist_val_loader, mnist_test_loader,
          **kwargs_train)
    
    # fonts_train_loader = torch.utils.data.DataLoader(
    #     FontData(mnist_train_data.dataset.data.numpy()),
    #     batch_size=10, shuffle=True)

    # Combine with font-generated data
    # TODO: NORMALISE PADDING?
    # Make sure there is enough data augmentation
    # Train while keeping track of training and validation loss for plots!
    # Plot a few numbers with their classification scores at the end
    # Don't forget batch normalisation (amplitude and padding)
    # Do a lot of augmentation on the PC numbers to match the size of the mnist dataset?
    # Careful, where are the zeros?


if __name__ == '__main__':
    main()
