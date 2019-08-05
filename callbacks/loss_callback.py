import os

import matplotlib.pyplot as plt
from keras.callbacks import Callback


class SaveLoss(Callback):
    """
    Save the training loss in a csv file and plot a figure.
    """
    def __init__(self, folder, scale='linear'):
        super(SaveLoss, self).__init__()
        self.folder = folder
        self.values = dict()
        self.scale = scale

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return

        if len(self.values) == 0:
            for k in logs:
                self.values[k] = []

        for k in logs:
            self.values[k].append(logs[k])

        plt.figure()
        plt.suptitle('Training loss', fontsize=16)
        for k in self.values:
            if 'dis' in k or 'adv' in k:
                continue

            epochs = range(len(self.values[k]))
            if self.scale == 'linear':
                plt.plot(epochs, self.values[k], label=k)
            elif self.scale == 'log':
                plt.semilogy(epochs, self.values[k], label=k)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.folder, 'training_loss.png'))

        plt.figure()
        plt.suptitle('Training loss', fontsize=16)
        for k in self.values:
            if not ('dis' in k or 'adv' in k):
                continue

            epochs = range(len(self.values[k]))
            plt.plot(epochs, self.values[k], label=k)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.folder, 'training_discr_loss.png'))

        plt.close()