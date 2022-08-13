"""Run a grid search on hyperparameters for network_v3."""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import itertools
# Project imports
from network_v3 import trainWrapper
import utils


def fitFunc_simple(x, C, D, x0):
    return C / (x - x0) + D


def fitFunc_expo(x, A, B, C):
    return np.exp(-A*x + B) + C


def getP0_expo(x, y):
    x1, x2 = x[0], x[-1]
    y1, y2 = y[0], y[-1]
    
    A = (np.log(y2) - np.log(y1)) / (x1 - x2)
    B = np.log(y1) + A*x1
    return A, B, 0


def getP0(x, y):
    x1, x2 = x[0], x[-1]
    y1, y2 = y[0], y[-1]

    D = min(y)
    x0 = (D*(x2 - x1) + y1*x1 - y2*x2) / (y1 - y2)
    C = (y1 - D)*(x1 - x0)
    return C, D, x0


def main():
    batch_size_train = 64
    kwargs_train = dict(
        kernel_size=3, nConvLayers=2, fcn_mid=50, nChannels=10,  # Train
        epochs=100, patience=5, lr_factor=.1, momentum=.9, lr=.01,  # Model
        dropout=.2)

    # Define values to explore in grid search
    nLs = 2, 3
    fcn_mids = 100, 150, 200, 300, 500
    
    # Use itertools to easily perform said grid-search
    iterables = nLs, fcn_mids
    combos = itertools.product(*iterables)
    combos_idx = itertools.product(*list(map(lambda x: np.arange(len(x)),
                                             iterables)))
    
    dataFit = np.zeros(tuple(map(len, iterables)))
    dataMin = np.zeros_like(dataFit)
    for (nL, fcn_mid), idx in zip(combos, combos_idx):
        # # Loop information
        kwargs_grid = dict(nConvLayers=nL, fcn_mid=fcn_mid)
        _str = "[GRIDSEARCH] Launch with:"
        for arg, val in zip(kwargs_grid.keys(),
                            kwargs_grid.values()):
            _str += "\n\t{}: {}".format(arg, val)
        print(_str)

        # Check if params are valid
        M = utils.getOutSize(28, kwargs_train["kernel_size"], 2, nL)
        if M <= 0:
            print("Invalid conv params, skipping..")
            continue

        # Evaluate grid point
        kwargs_train.update(kwargs_grid)
        val_loss = trainWrapper(batch_size_train, **kwargs_train,
                                verbose=False)
        dataMin[idx] = min(val_loss)
        np.save("data_min.npy", dataMin)
        
        # ######
        # np.save("tmp_loss.npy", val_loss)
        # val_loss = np.load("tmp_loss.npy", allow_pickle=True)
        # ######
        
        # Plot
        plt.figure()
        ax = plt.subplot(111)
        ax.set_yscale("log")
        ax.plot(val_loss)

        # Try simple fit
        x = np.arange(len(val_loss))
        try:
            popt_s, pcov_s = curve_fit(
                fitFunc_simple, x, val_loss,
                p0=getP0(x, val_loss))
            chi_simple = utils.getChi(x, val_loss,
                                      np.ones(len(x)),
                                      fitFunc_simple, popt_s)
            ax.plot(x, fitFunc_simple(x, *popt_s),
                    label="{:.2e}".format(chi_simple))
        except Exception:
            print("Simple fit failed")
            chi_simple = np.inf
        
        # Try expo fit
        try:
            xFit, y = x[-10:], val_loss[-10:]
            popt, pcov = curve_fit(fitFunc_expo, xFit, y,
                                   p0=getP0_expo(xFit, y))
            chi = utils.getChi(xFit, y, np.ones(len(xFit)),
                               fitFunc_expo, popt)
            ax.plot(xFit, fitFunc_expo(xFit, *popt),
                    label="{:.2e}".format(chi))
        except Exception:
            print("Expo fit failed")
            chi = np.inf
            if np.isinf(chi_simple):
                print("both fits failed.. skipping point")
                dataFit[idx] = np.nan
                continue
    
        # Nice things
        ax.legend(loc="upper right")
        
        # Use result with best chi^2
        if chi < chi_simple:  # ok cause sigma not norm
            f, popt = fitFunc_expo, popt
        else:
            f, popt = fitFunc_simple, popt_s
            
        # Return extrapolated function at epoch 100
        dataFit[idx] = f(np.array([100]), *popt)
        np.save("data_fit.npy", dataFit)
        ax.set_title("Extrapolated loss: {:.2e}".format(dataFit[idx]))
        x = np.arange(x[-10], 101)
        ax.plot(x, f(x, *popt), linestyle="--", color="C2")
        plt.show()


def eval():
    dataFit = np.load("data_fit.npy")
    dataMin = np.load("data_min.npy")
    
    for data in dataFit, dataMin:
        data[data == 0] = np.nan
        data[np.isnan(data)] = np.inf
        
        min_x = np.inf
        for idx, x in np.ndenumerate(data):
            if x < min_x:
                min_x = x
                min_idx = idx
        print(min_idx)
        print(min_x)


if __name__ == '__main__':
    eval()
    