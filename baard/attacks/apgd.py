"""
A wrapper for Auto Projected Gradient Descent (APGD) attack using the Trusted-AI `adversarial-robustness-toolbox`.
URL for the package: https://github.com/Trusted-AI/adversarial-robustness-toolbox/
"""
from typing import Any, Union

import numpy as np
import torch
from art.attacks.evasion import AutoProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from torch import Tensor
from torch.nn import Module


def auto_projected_gradient_descent(
    model_fn: Module,
    x: Tensor,
    norm: Union[float, int] = np.inf,
    n_classes: int = 10,
    eps: float = 0.3,
    eps_iter: float = 0.1,
    nb_iter: int = 100,
    nb_rand_init: int = 5,
    clip_min: float = 0,
    clip_max: float = 1,
    device: str = 'gpu',
    y: Tensor = None,
    targeted: bool = False,
    loss: Any = None,
    optimizer: Any = None,
) -> Tensor:
    """APGD wrapper for ART AutoProjectedGradientDescent. 
    URL: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/auto_projected_gradient_descent.py
    """

    if loss is None:
        loss = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model_fn.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    device_type = 'cpu' if device == 'cpu' or not torch.cuda.is_available() else 'gpu'
    input_shape = tuple(x.size()[1:])
    batch_size = x.size()[0]  # The x pass into the function is the mini batch.

    classifier = PyTorchClassifier(
        model=model_fn,
        loss=loss,
        input_shape=input_shape,
        optimizer=optimizer,
        nb_classes=n_classes,
        clip_values=(clip_min, clip_max),
        device_type=device_type,
    )
    attack = AutoProjectedGradientDescent(
        estimator=classifier,
        norm=norm,
        eps=eps,
        eps_step=eps_iter,
        max_iter=nb_iter,
        targeted=targeted,
        nb_random_init=nb_rand_init,
        batch_size=batch_size,
        verbose=False,
    )
    # ART only takes Numpy Array.
    x_np = x.cpu().detach().numpy()
    if y is not None:
        y = torch.nn.functional.one_hot(y)
        y = y.cpu().detach().numpy()
    adv_np = attack.generate(x=x_np, y=y)
    adv_torch = torch.from_numpy(adv_np)
    return adv_torch
