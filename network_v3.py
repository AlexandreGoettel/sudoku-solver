"""Train MNIST based on https://nextjournal.com/gkoehler/pytorch-mnist."""
# Standard imports
import numpy as np
from matplotlib import pyplot as plt
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
    
    def __init__(self, kernel_size=5, dropout=.3):
        super(ImageClassifier, self).__init__()
        # Hyper-params
        self.dropout = dropout
        # Network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(train_loader, val_loader, test_loader,
          dropout=.3, lr=.001, momentum=.9, lr_factor=.1, epochs=100):
    """Train the "ImageClassifier" using the passed datasets."""
    model = ImageClassifier(dropout=dropout)
    print(summary(model, (64, 1, 28, 28)))


def main():
    # Set hyperparameters
    batch_size_train = 64
    validation_size = 5000

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
    
    train(mnist_train_loader, mnist_val_loader, mnist_test_loader)
    
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

"""
# Training loop
for epoch in range(epochs):
    print("Epoch: {:d}".format(epoch+1))
    plot_data[epoch, ...] = 0.
    for batch in range(nBatches):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Training step
        data, labels = data_train[batch, ...], labels_train[batch, :]
        output = model(data)
        loss = criterion(output, labels)
        
        # Update weights
        loss.backward()
        optimizer.step()
        
        # Book-keeping
        output = output.detach().numpy()
        acc, sig = getAccuracy(output, labels.numpy())
        # TODO: plot this from validation data not training..
        plot2_data[epoch, ...] += output[...] / nBatches
        plot_data[epoch, 0] += loss.item() / nBatches
        plot_data[epoch, 2:4] += acc / nBatches, sig / nBatches
        
        # Printing and plotting
        print("Batch {:d}/{:d}".format(batch+1, nBatches))
        print("\tTraining loss: {:.4f}".format(loss.item()))
        print("\tTraining accuracy: {:.3f} +- {:.4f}".format(acc, sig))
        
    # Validation
    model.eval()
    output = model(data_val)
    loss = criterion(output, labels_val)
    
    # Learning rate scheduler
    # scheduler.step(loss)
    
    # Book-keeping
    acc, sig = getAccuracy(output.detach().numpy(), labels_val.numpy())
    plot_data[epoch, 1] = loss.item()
    plot_data[epoch, 4:6] = acc, sig
    print("Validation loss: {:.4f}".format(plot_data[epoch, 1]))
    print("Validation accuracy: {:.3f} +- {:.4f}".format(acc, sig))
    
    if epoch < 2:
        continue
    
    # Plotting
    # Output
    fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
    count = 0
    for i in range(4):
        for j in range(4):
            ax[i][j].imshow(plot2_data[:epoch+1, :, count],
                            extent=(0, 15, 15, 0), interpolation="None")
            count += 1

    # Loss
    plt.figure()
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.plot(plot_data[:, 0], label="Training")
    ax.plot(plot_data[:, 1], label="Validation")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    # ax.axhline(2.7726, color="C3")
    # ax.axhline(1.8746, color="C2")
    
    # Accuracy
    plt.figure()
    plt.errorbar(x_epochs, plot_data[:, 2], yerr=plot_data[:, 3], fmt=".",
                 label="Training")
    plt.errorbar(x_epochs, plot_data[:, 4], yerr=plot_data[:, 5], fmt=".",
                 label="Validation")
    plt.legend(loc="best")
    plt.ylabel("Accuracy")
    plt.axhline(1/16., linestyle="--", linewidth=2., color="r")
    plt.show()

    # Save model progress
    torch.save(model.state_dict(), "Marlene_test1.pth")
"""