from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import config as cf

files = glob("results/*.npy")
files = sorted(files, key=lambda x: int(x.split(' - ')[1]))

epochs = []
for file in files:
    epochs.append(np.load(file))
epochs = np.swapaxes(np.array(epochs), 0, 1)
x = range(epochs.shape[1])
for i in range(10):
    plt.plot(epochs[i, :, 0], epochs[i, :, 1], label=cf.classes[i])
    plt.text(epochs[i, -1, 0], epochs[i, -1, 1], cf.classes[i])
x_min = epochs[:, :, 0].min()
x_max = epochs[:, :, 0].max()
y_min = epochs[:, :, 1].min()
y_max = epochs[:, :, 1].max()
plt.ylim([y_min - 1, y_max + 1])
plt.xlim([x_min - 1, x_max + 1])
plt.title("PCA Embedding over time of representation layer")
plt.savefig("PCA over time.png")
