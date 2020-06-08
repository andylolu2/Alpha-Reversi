import matplotlib as mpl

mpl.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = "Trial8"
training_results_dir = "training_results\\{}_with_L2.npz".format(MODEL_NAME)
training_results = np.load(training_results_dir)
train_loss_results = training_results["loss"].tolist()
train_mse_results = training_results["mse"].tolist()
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_ylim([0, 1.4])
axes[0].plot(train_loss_results)

axes[1].set_ylabel("MSE", fontsize=14)
axes[1].set_ylim([0, None])
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_mse_results)
plt.show()
