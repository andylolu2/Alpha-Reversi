import matplotlib as mpl

mpl.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = "Trial6"
training_results_dir = "training_results\\{}.npz".format(MODEL_NAME)
training_results = np.load(training_results_dir)
train_loss_results = training_results["loss"].tolist()
fig, axes = plt.subplots(1, sharex=True, figsize=(12, 5))
fig.suptitle('Training loss', fontsize=20)

axes.set_ylabel("Loss", fontsize=14)
axes.set_ylim([0.5, 1.3])
axes.set_xlabel("Train steps", fontsize=14)
axes.plot(train_loss_results)
# plt.show()

fig.savefig("training_loss.png")
