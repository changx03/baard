"""
This code is based on cleverhans-lab/cleverhans repository.
Link here: `https://github.com/cleverhans-lab/cleverhans`,
accessed  on 27-Oct-2022.

White-box adaptive attack on BAARD.
"""
import logging
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from .utils import clip_eta, optimize_linear

logger = logging.getLogger(__name__)


def targeted_whitebox_pgd(model: Module,
                          X: Tensor,
                          X_target: Tensor,
                          eps: float,
                          eps_iter: float,
                          nb_iter: int = 100,
                          norm: Union[float, int] = np.inf,
                          clip: tuple[float, float] = (0., 1.),
                          rand_init: bool = True,
                          c: float = 1.,
                          early_stop: bool = False,
                          ) -> Tensor:
    """
    Apply targeted adaptive white-box PGD attack on BAARD detector.

    :param model: a callable that takes an input tensor and returns the model logits.
    :param X: input tensor.
    :param X_target: targeted X.
    :param eps: Epsilon. To control the perturbation.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations. Default is 100.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        Default is np.inf.
    :param clip: tuple[float, float]. Minimum and maximum float value for
        adversarial example components. Default is (0, 1).
    :param rand_init: (optional) bool. Whether to start the attack from a
        randomly perturbed x.
    :param c: the parameter that controls the weight of how X close to X_target.
    :param early_stop: bool. Stops early when predictions match targets.
    :return: a tensor for the adversarial example.
    """
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if len(clip) != 2 or clip[0] > clip[1]:
        raise ValueError(f"Clip range must bt (min, max). Got ({clip}).")

    # Initialize loop variables
    if rand_init:
        eta = torch.zeros_like(X).uniform_(-eps, eps)
    else:
        eta = torch.zeros_like(X)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    X_adv = X + eta
    X_adv = torch.clamp(X_adv, clip[0], clip[1])

    # NOTE: Try to match the output of the target!
    # y_target = torch.argmax(model(X_target), 1)
    y_target = model(X_target).detach().to(torch.float).requires_grad_(False)

    # loss_fn1 = torch.nn.CrossEntropyLoss()
    loss_fn1 = torch.nn.MSELoss()
    loss_fn2 = torch.nn.MSELoss()

    i = 0
    while i < nb_iter:
        # Create a copy with gradient.
        X_next = X_adv.clone().detach().to(torch.float).requires_grad_(True)

        # NOTE: Negative, because it's targeted!
        loss1 = -loss_fn1(model(X_next), y_target)
        loss2 = -loss_fn2(X_next, X_target)
        loss = loss1 + (c * loss2)
        loss.backward()
        optimal_perturbation = optimize_linear(X_next.grad, eps_iter, norm)
        X_adv = X_next + optimal_perturbation

        # Clipping perturbation eta to norm norm ball
        eta = X_adv - X
        eta = clip_eta(eta, norm, eps)
        X_adv = X + eta

        # Redo the clipping.
        X_adv = torch.clamp(X_adv, clip[0], clip[1])
        i += 1

        # Early stopping to target.
        if early_stop:
            y_adv = torch.argmax(model(X_adv), 1)
            if torch.all(y_adv == y_target):
                logger.info('Meet target at %d iteration. Stop!', i)
                break
    return X_adv
