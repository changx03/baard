"""
This code is based on cleverhans-lab/cleverhans repository.
Link here: `https://github.com/cleverhans-lab/cleverhans`,
accessed  on 27-Oct-2022.

White-box adaptive attack on BAARD.
"""
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from .utils import clip_eta, optimize_linear


def targeted_whitebox_pgd(model: Module,
                          x: Tensor,
                          y: Tensor,
                          eps: float,
                          eps_iter: float,
                          nb_iter: int = 100,
                          norm: Union[float, int] = np.inf,
                          clip: tuple[float, float] = (0., 1.),
                          rand_init: bool = True,
                          ) -> Tensor:
    """
    Apply targeted adaptive white-box PGD attack on BAARD detector.

    :param model: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param y: Tensor with  the target label.
    :param eps: Epsilon. To control the perturbation.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations. Default is 100.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
              Default is np.inf.
    :param clip: tuple[float, float]. Minimum and maximum float value for 
                 adversarial example components. Default is (0, 1).
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :return: a tensor for the adversarial example.
    """
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if len(clip) != 2 or clip[0] > clip[1]:
        raise ValueError(f"Clip range must bt (min, max). Got ({clip}).")

    # Initialize loop variables
    if rand_init:
        eta = torch.zeros_like(x).uniform_(-eps, eps)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    adv_x = torch.clamp(adv_x, clip[0], clip[1])

    # NOTE: This adaptive attack is targeted!
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model(x), 1)

    i = 0
    while i < nb_iter:
        # Create a copy with gradient.
        x_next = adv_x.clone().detach().to(torch.float).requires_grad_(True)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(model(x_next), y)
        if y is not None:  # Targeted
            loss = -loss
        loss.backward()
        optimal_perturbation = optimize_linear(x_next.grad, eps_iter, norm)
        adv_x = x_next + optimal_perturbation

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        adv_x = torch.clamp(adv_x, clip[0], clip[1])
        i += 1

    return adv_x
