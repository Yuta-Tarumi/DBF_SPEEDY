import torch
import matplotlib.pyplot as plt

data = torch.load("Lorenz96_output/epoch_001/batch_0000.pt")
fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_subplot(131)
ax1.imshow(data["observations"][0], vmin=0, vmax=10)
ax2 = fig.add_subplot(132)
ax2.imshow(data["targets"][0], vmin=-5, vmax=10)
ax3 = fig.add_subplot(133)
ax3.imshow(data["reconstruction"][0], vmin=-5, vmax=10)
ax1.set_title("obs")
ax2.set_title("true")
ax3.set_title("model")
fig.suptitle("Lorenz96")

plt.savefig("example_Lorenz")
