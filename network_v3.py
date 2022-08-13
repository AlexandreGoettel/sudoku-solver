"""Train MNIST based on https://nextjournal.com/gkoehler/pytorch-mnist."""
# Standard imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
# Torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
# Project imports
import utils
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
        self.kernel_size = kernel_size
        # Network
        for n in range(n_layers):
            setattr(self, "conv{}".format(n+1), nn.Conv2d(
                max(1, n*channels_per_layer), (n+1)*channels_per_layer,
                kernel_size=kernel_size, stride=1, padding="valid"))
        
        self.M = utils.getOutSize(28, kernel_size, 2, n_layers)**2 *\
            n_layers*channels_per_layer
        self.fc1 = nn.Linear(self.M, fcn_mid)
        self.fc2 = nn.Linear(fcn_mid, 10)

    def forward(self, x):
        for n in range(self.n_layers):
            layer = getattr(self, "conv{}".format(n+1))(x)
            m = utils.getOutSize(x.shape[-1], self.kernel_size, 2, 1,
                                 raw=True)
            x = F.leaky_relu(F.max_pool2d(layer, 2, padding=m%2))
        
        x = x.view(-1, self.M)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc2(x)
        return F.log_softmax(x, -1)


def batch_normalise(data):
    nBatch = data.shape[0]
    numpy_data = np.sum(data.detach().numpy(), axis=0)

    mu = np.mean(numpy_data) / nBatch
    sigma = np.std(numpy_data.flatten(), ddof=1)
    data = (data - mu) / sigma
    return data


def train(train_loader, val_loader, test_loader, model_path="mnist_test1.pth",
          dropout=.3, lr=.001, momentum=.9, lr_factor=.1, epochs=10,
          patience=5, kernel_size=5, nConvLayers=2, fcn_mid=50, nChannels=10,
          verbose=False):
    """Train the "ImageClassifier" using the passed datasets."""
    model = ImageClassifier(dropout=dropout, kernel_size=kernel_size,
                            n_layers=nConvLayers, fcn_mid=fcn_mid,
                            channels_per_layer=nChannels)
    print(summary(model, input_size=(64, 1, 28, 28)))

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
            train_accuracy[:, ibatch] = utils.getAccuracy(output,
                                                          truth.numpy())
            train_loss[ibatch] = loss.item()

        book[epoch, 0, :2] = utils.weighedAverage(*train_accuracy)
        book[epoch, 0, 2] = np.mean(train_loss)

        print("[TRAIN] acc: {:.3f} +- {:.3f}\n[TRAIN] loss: {:.4f}".format(
            *book[epoch, 0, :]))

        # Switch to validation mode
        model.eval()
        with torch.no_grad():
            for ibatch, (data, truth) in enumerate(val_loader):
                data = batch_normalise(data)
                output = model(data)
                loss = criterion(output, truth)
    
                # Book-keeping
                output = output.detach().numpy()
                val_accuracy[:, ibatch] = utils.getAccuracy(output, truth.numpy())
            val_loss[ibatch] = loss.item()
        book[epoch, 1, :2] = utils.weighedAverage(*val_accuracy)
        book[epoch, 1, 2] = np.mean(val_loss)
        print("[VALID] acc: {:.3f} +- {:.3f}\n[VALID] loss: {:.4f}".format(
            *book[epoch, 1, :]))

        # Update scheduler
        scheduler.step(book[epoch, 1, 2])

        # Save model progress
        torch.save(model.state_dict(), model_path)

        # Update plots
        if not verbose or epoch < 2:
            continue
        fig = plt.figure()
        gs = GridSpec(2, 3, figure=fig)
        axAcc, axLoss = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
        axLoss.set_yscale("log")

        x = np.arange(epoch+1)
        labels = ["Training", "Validation"]
        for i in range(2):
            axAcc.errorbar(x+i*.1+1, book[:epoch+1, i, 0],
                           book[:epoch+1, i, 1],
                           fmt=".", label=labels[i], color="C{}".format(i))
            axLoss.plot(x+1, book[:epoch+1, i, 2], "-",
                        label=labels[i], color="C{}".format(i))
        axLoss.axhline(min(book[:epoch+1, 1, 2]), color="C2", linestyle="--",
                       label="min val loss")

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
        test_accuracy[:, ibatch] = utils.getAccuracy(output, truth.numpy())
        test_loss[ibatch] = loss.item()
    results = np.zeros(3)
    results[:2] = utils.weighedAverage(*val_accuracy)
    results[2] = np.mean(test_loss)
    print("[TEST] acc: {:.3f} +- {:.3f}\n[TEST] loss: {:.4f}".format(
        *results))
    
    return book[:, 1, 2]  # validation loss


def trainWrapper(batch_size_train=64, verbose=False, **kwargs_train):
    """Do the training on mnist."""
    # Set hyperparameters
    validation_size = 5000

    # Great model saved under "mnist_test1.pth"
    kwargs_train = dict(
        kernel_size=3, nConvLayers=3, fcn_mid=250, nChannels=10,  # Train
        epochs=500, patience=75, lr_factor=.1, momentum=.9, lr=.01,  # Model
        dropout=.2)  # Model

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

    return train(mnist_train_loader, mnist_val_loader, mnist_test_loader,
                 verbose=verbose, model_path="playing.pth", **kwargs_train)


def trainPC():
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
    pass


if __name__ == '__main__':
    # trainWrapper(verbose=True)
    trainPC()
