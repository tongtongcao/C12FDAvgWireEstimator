import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm

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
        outname = f"{self.print_dir}/loss_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plot_diff(self, preds, targets):
        plt.figure(figsize=(20, 20))
        diff = preds - targets

        # Plot histogram with real counts
        counts, bins, _ = plt.hist(diff, bins=1000, density=False)
        plt.xlim(-5, 5)
        plt.xlabel("Diff")
        plt.ylabel("Counts")
        plt.title("Diff. between prediction and target")

        # Gaussian fit in restricted window
        fit_min, fit_max = -0.5, 0.5
        mask = (diff >= fit_min) & (diff <= fit_max)
        diff_fit = diff[mask]
        mu, std = norm.fit(diff_fit)

        # Scale Gaussian to histogram counts
        bin_width = bins[1] - bins[0]
        x = np.linspace(bins[0], bins[-1], 500)
        p = norm.pdf(x, mu, std) * len(diff_fit) * bin_width

        plt.plot(x, p, 'r', linewidth=2)
        plt.text(
            0.95, 0.95,
            f"$\\mu = {mu:.2f}$\n$\\sigma = {std:.2f}$",
            transform=plt.gca().transAxes,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
        )

        outname = f"{self.print_dir}/diff_{self.end_name}.png"
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
        outname = f"{self.print_dir}/pred_target_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()