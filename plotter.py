import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.rcParams.update({
    'font.size': 40,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 15,
    'xtick.minor.size': 10,
    'ytick.major.size': 15,
    'ytick.minor.size': 10,
    'xtick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.major.width': 3,
    'ytick.minor.width': 3,
    'axes.linewidth': 3,
    'figure.max_open_warning': 200,
    'lines.linewidth': 5
})


class Plotter:
    def __init__(self, print_dir='', end_name=''):
        self.print_dir = print_dir
        self.end_name = end_name

    def plotTrainLoss(self, tracker):
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        plt.figure(figsize=(20, 20))
        plt.plot(train_losses, label='Train', color='royalblue')
        plt.plot(val_losses, label='Test', color='firebrick')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.print_dir}loss{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plot_diff(self, preds, targets):
        plt.figure(figsize=(20, 20))
        plt.hist(preds - targets, bins = 100)
        plt.xlim(-5, 5)
        plt.xlabel("Diff")
        plt.ylabel("Counts")
        plt.title("Diff. between prediction and target")
        outname = f"{self.print_dir}diff{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plot_pred_target(self, preds, targets):
        plt.figure(figsize=(20, 20))
        plt.hist2d(targets, preds, bins=(300, 300), range=[[0, 112], [0, 112]], cmap='viridis', norm=LogNorm())
        plt.colorbar(label='Counts')
        plt.xlabel("target")
        plt.ylabel("pred")
        plt.title("prediction vs target")
        plt.plot([0, 112], [0, 112], 'r--', label='Ideal Prediction') # diagonal line
        plt.legend()
        outname = f"{self.print_dir}pred_target{self.end_name}.png"
        plt.savefig(outname)
        plt.close()