# %%
import os
from pathlib import Path

from baard.utils.torch_utils import plot_images

PATH_ROOT = Path(os.getcwd()).absolute().parent
print(PATH_ROOT)

# %%
path_clean = os.path.join(PATH_ROOT, 'results', 'exp366364', 'MNIST')
plot_images(path_clean, 'L2', [4, 8, 12], 'APGD', n=1000)

# %%
plot_images(path_clean, 'Linf', [0.22, 0.66], 'APGD', n=1000)
# %%
path_clean = os.path.join(PATH_ROOT, 'results', 'exp366364', 'CIFAR10')
plot_images(path_clean, 'L2', [0.3, 1.8, 3], 'APGD', n=1000)

# %%
plot_images(path_clean, 'Linf', [0.01, 0.1, 0.3], 'APGD', n=1000)
