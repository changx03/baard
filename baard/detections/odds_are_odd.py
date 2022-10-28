"""Implementing the paper "THe odds are odd" detector."""
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm

from baard.utils.torch_utils import batch_forward, create_noisy_examples


def get_negative_labels(y, n_classes=10):
    """Get every label except the true label."""
    labels = np.arange(n_classes)
    return labels[labels != y]


def compute_single_noise_alignment(hidden_out_x: Tensor, hidden_out_noise: Tensor,
                                   negative_labels, weight_relevant: Tensor
                                   ) -> Tensor:
    """Compute the odds on noisy examples (1 noise type) for a single input."""
    hidden_output_diff = hidden_out_x - hidden_out_noise
    odds = torch.matmul(hidden_output_diff, weight_relevant.transpose(1, 0))   # Size: [n_samples, n_classes]
    odds = odds[:, negative_labels]  # Size: [n_samples, n_classes-1]
    return odds


def compute_multiple_noise_alignments(latent_net: Module, x: Tensor, pred: int,
                                      weight: Tensor, noise_list: List,
                                      noise_clip_range: Tuple = (0, 1),
                                      n_samples: int = 30, n_classes: int = 10,
                                      ) -> OrderedDict:
    """Compute all odds for every noise on a single input."""
    if isinstance(pred, Tensor):
        pred = int(pred.item())

    hidden_out_x = batch_forward(latent_net, x.unsqueeze(0))
    weight_diff = weight.unsqueeze(0) - weight.unsqueeze(1)
    weight_relevant = weight_diff[:, pred]
    negative_labels = get_negative_labels(pred, n_classes)
    alignments = OrderedDict()
    for noise_eps in noise_list:
        x_noisy = create_noisy_examples(x, n_samples, noise_eps, clip_range=noise_clip_range)
        hidden_out_noise = batch_forward(latent_net, x_noisy)
        odds = compute_single_noise_alignment(hidden_out_x, hidden_out_noise,
                                              negative_labels, weight_relevant)
        alignments[noise_eps] = odds
    return alignments


def collect_weights_stats(latent_net: Module, X: Tensor, preds: Tensor,
                          weight: Tensor, noise_list: List,
                          noise_clip_range: Tuple = (0, 1), n_samples: int = 30,
                          n_classes: int = 10
                          ) -> Dict:
    """Collect weights stats from a dataset."""
    weights_stats = {(c, noise_eps): [] for c in range(n_classes) for noise_eps in noise_list}

    with torch.no_grad():
        pbar = tqdm(zip(X, preds), total=len(X))
        pbar.set_description('Collecting weight stats', refresh=False)
        for _x, _y in pbar:
            label = int(_y.item())
            alignments = compute_multiple_noise_alignments(
                latent_net, _x, pred=label, weight=weight, noise_list=noise_list,
                noise_clip_range=noise_clip_range, n_samples=n_samples,
                n_classes=n_classes)
            for noise_eps in alignments:
                weights_stats[(label, noise_eps)].append(alignments[noise_eps])

    for key in weights_stats:
        weights = torch.vstack(weights_stats[key])
        # Replace the weights with mean and std.
        weights_stats[key] = {'mean': weights.mean(0), 'std': weights.std(0)}

    return weights_stats


def detect_single_example(weights_stats: Dict, latent_net: Module, x: Tensor,
                          pred: int, weight: Tensor, noise_list: List,
                          noise_clip_range: Tuple = (0, 1), n_samples: int = 30,
                          n_classes: int = 10):
    """Detect single example. Return the absolute maximum Z-score."""
    if isinstance(pred, Tensor):
        pred = int(pred.item())

    with torch.no_grad():
        alignments = compute_multiple_noise_alignments(
            latent_net, x, pred=pred, weight=weight, noise_list=noise_list,
            noise_clip_range=noise_clip_range, n_samples=n_samples,
            n_classes=n_classes)

        max_Z_scores = []
        for noise_eps in noise_list:
            score = alignments[noise_eps].mean(0)
            mean = weights_stats[(pred, noise_eps)]['mean']
            std = weights_stats[(pred, noise_eps)]['std']
            z_score = (score - mean) / std
            # 2 tailed Z-score
            z_max = np.abs(z_score.detach().numpy()).max()
            max_Z_scores.append(z_max)

    return np.max(max_Z_scores)


class OddsAreOddDetector:
    """Implement Odds are odd detector in PyTorch."""

    def __init__(self) -> None:
        pass

    def train(self):
        # weights_stats = collect_weights_stats(
        #     latent_net, X_test, preds=Y_test,
        #     weight=weight, noise_list=noise_list, n_samples=n_samples,
        # )
        pass

    def evaluate(self):
        # z_max = detect_single_example(weights_stats, latent_net, x=X_unused[0],
        #                               pred=Y_unused[0].item(), weight=weight, noise_list=noise_list,
        #                               n_samples=n_samples)
        pass


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    from baard.classifiers.mnist_cnn import MNIST_CNN

    # # Test create_noisy_examples
    # x = torch.zeros(1, 28, 28)
    # x_u = create_noisy_examples(x, n_samples=8, noise_eps='u0.01')
    # assert x_u.size() == (8, 1, 28, 28)
    # assert not x.equal(x_u[0])

    # # Test get_negative_labels
    # negative_labels = get_negative_labels(3, 5)
    # assert np.all(negative_labels == [0, 1, 2, 4])

    # Test compute_multiple_noise_alignments

    n_classes = 10
    model = MNIST_CNN()
    weight = list(model.children())[-1].weight
    loader = model.val_dataloader()
    X, Y = next(iter(loader))
    X_unused, X_test, Y_unused, Y_test = train_test_split(X, Y, test_size=30, random_state=0)
    assert len(set(Y_test.tolist())) == 10, 'Missing labels!'
    print('Y_test:', Y_test)

    x = X[0]  # Only need 1 example.
    assert x.size() == (1, 28, 28)
    y = Y[0].item()
    latent_net = torch.nn.Sequential(
        model.conv1,
        model.relu1,
        model.conv2,
        model.relu2,
        model.pool1,
        model.flatten,
        model.fc1,
    )

    # # Test batch_forward
    # outputs = batch_forward(latent_net, X, batch_size=32, device='cpu')
    # assert outputs.size() == (len(X), 200)

    # n_samples = 128
    # noise_list = ['n0.01', 's0.01', 'u0.01']
    # alignments = compute_multiple_noise_alignments(
    #     latent_net,
    #     x,
    #     y,
    #     weight,
    #     noise_list,
    #     n_samples=n_samples
    # )
    # for noise in noise_list:
    #     assert alignments[noise].size() == (n_samples, n_classes - 1)

    noise_list = ['n0.01', 'u0.01']
    n_samples = 20
    weights_stats = collect_weights_stats(
        latent_net, X_test, preds=Y_test,
        weight=weight, noise_list=noise_list, n_samples=n_samples,
    )

    # Test detect_single_example
    z_max = detect_single_example(weights_stats, latent_net, x=X_unused[0],
                                  pred=Y_unused[0].item(), weight=weight, noise_list=noise_list,
                                  n_samples=n_samples)
    print('z_max:', z_max)
