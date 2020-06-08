import matplotlib as mpl
from model_path_management import *
mpl.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

target = np.load(get_elo_rating_dir(model_elo_rating_path))
models_elo = target["models"].tolist()
best_models_elo = target["best_models"].tolist()
fig, axes = plt.subplots(2, sharex=False, figsize=(8, 8))
fig.suptitle('Models', fontsize=20)

axes[0].set_ylabel("Elo", fontsize=14)
axes[0].set_ylim([0, max(models_elo)*1.1])
axes[0].set_xlabel("Models version", fontsize=14)
axes[0].set_xlim([0, len(models_elo)])
axes[0].plot(models_elo)

axes[1].set_ylabel("Elo", fontsize=14)
axes[1].set_ylim([0, max(best_models_elo)*1.1])
axes[1].set_xlim([0, len(best_models_elo)])
axes[1].set_xlabel("Best models version", fontsize=14)
axes[1].plot(best_models_elo)
plt.show()

