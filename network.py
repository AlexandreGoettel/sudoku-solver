"""Train MNIST based on https://nextjournal.com/gkoehler/pytorch-mnist."""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt


class imageClassifier(nn.Module):
    
    def __init__(self, kernel_size=5):
        super(imageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def main():
    # Set parameters for the training
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    
    # Get DataLoaders for the training
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("mnist/", train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))])),
        batch_size=batch_size_train, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("mnist/", train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))])),
        batch_size=batch_size_test, shuffle=True)

    # Prepare training
    train_losses, train_count = [], []
    
    # Initialise
    network = imageClassifier()
    optimiser = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    
    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimiser.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimiser.step()
            
            # Log information
            if not batch_idx % log_interval:
                print("Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx*len(data), len(train_loader.dataset),
                    100.*batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_count.append(batch_idx*batch_size_train +
                                ((epoch-1)*len(train_loader.dataset)))
                torch.save(network.state_dict, "results/model.pth")
                torch.save(optimiser.state_dict, "results/optimiser.pth")
    
    def test():
        network.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target,
                                        size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        print("\nTest: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
            test_loss, correct, len(test_loader.dataset),
            100*correct / len(test_loader.dataset)))
    
    # Now perform the training!
    test()
    for epoch in range(n_epochs):
        train(epoch+1)
    test()
    
    # Plots
    plt.plot(train_count, train_losses)
    plt.ylabel("-log lkl loss")
    plt.xlabel("Number of examples seen")
    

if __name__ == '__main__':
    main()
    
                