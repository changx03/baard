"""Implementing the paper "THe odds are odd" detector."""
import os
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.utils.torch_utils import (batch_forward, create_noisy_examples,
                                     dataloader2tensor, get_correct_examples,
                                     get_dataloader_shape, predict)

DATASETS = ['MNIST', 'CIFAR10']


def get_negative_labels(y, n_classes=10):
    """Get every label except the true label."""
    labels = np.arange(n_classes)
    return labels[labels != y]


def get_odds_params_from_data(data_name: str, model: LightningModule):
    """Get Odds are odd parameters based on dataset."""
    if data_name == DATASETS[0]:  # MNIST
        latent_net = torch.nn.Sequential(
            model.conv1,
            model.relu1,
            model.conv2,
            model.relu2,
            model.pool1,
            model.flatten,
            model.fc1,
        )
        weight = list(model.children())[-1].weight
        n_classes = 10
        clip_range = (0, 1)
        return latent_net, weight, n_classes, clip_range
    elif data_name == DATASETS[1]:  # CIFAR10
        raise NotImplementedError()
    else:
        raise NotImplementedError()


class OddsAreOddDetector:
    """Implement Odds are odd detector in PyTorch."""

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 noise_list: List,
                 n_noise_samples: int = 100,
                 device: str = 'cuda',
                 ):
        self.model = model
        self.data_name = data_name
        self.noise_list = noise_list
        self.n_noise_samples = n_noise_samples
        self.device = device

        if not torch.cuda.is_available() and device == 'cuda':
            warnings.warn('GPU is not available. Using CPU...')
            device = 'cpu'

        # Parameters from LightningModule:
        self.batch_size = self.model.train_dataloader().batch_size
        self.num_workers = self.model.train_dataloader().num_workers

        # Parameters based on data:
        latent_net, weight, n_classes, clip_range = get_odds_params_from_data(data_name, model)
        self.latent_net = latent_net
        self.weight = weight
        self.n_classes = n_classes
        self.noise_clip_range = clip_range
        self.weight_diff = self.weight.unsqueeze(0) - self.weight.unsqueeze(1)

        # Tunable parameters:
        self.weights_stats = None

    def train(self, X: Tensor, y: Tensor):
        """Train detector."""
        # Check predictions and true labels
        assert len(X) == len(y)
        dataloader = DataLoader(TensorDataset(X, y),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        correct_dataloader = get_correct_examples(self.model, dataloader)
        x_corr_shape = get_dataloader_shape(correct_dataloader)
        if x_corr_shape[0] != len(X):
            warnings.warn(f'{len(X) - x_corr_shape.size(0)} are classified incorrectly! Use {len(x_corr_shape)} examples instead.')

        # only uses correctly classified examples.
        X, y = dataloader2tensor(correct_dataloader)

        # Train weights
        weights_stats = self.__collect_weights_stats(X, preds=y)
        self.weights_stats = weights_stats

    def extract_features(self, X: Tensor) -> Tensor:
        """Extract features from examples."""
        if self.weights_stats is None:
            raise Exception('Weights statistics have not initialized yet. Call `train` or `load_weights_stats` first!')
        n = X.size(0)
        dataloader = DataLoader(TensorDataset(X),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        preds = predict(self.model, dataloader)
        scores = np.zeros(n)
        pbar = tqdm(range(n), total=n)
        pbar.set_description('Odds inference time', refresh=False)
        for i in pbar:
            _x = X[i]
            _pred = preds[i]
            z_max = self.detect_single_example(_x, _pred)
            scores[i] = z_max
        return scores

    def detect_single_example(self, x: Tensor, pred: int):
        """Detect single example. Return the absolute maximum Z-score."""
        if isinstance(pred, Tensor):
            pred = int(pred.item())

        with torch.no_grad():
            alignments = self.__compute_multiple_noise_alignments(x, pred=pred)

            max_Z_scores = []
            for noise_eps in self.noise_list:
                score = alignments[noise_eps].mean(0)
                mean = self.weights_stats[(pred, noise_eps)]['mean']
                std = self.weights_stats[(pred, noise_eps)]['std']
                z_score = (score - mean) / std
                # 2 tailed Z-score
                z_max = np.abs(z_score.detach().numpy()).max()
                max_Z_scores.append(z_max)

        return np.max(max_Z_scores)

    def save_weights_stats(self, path_output: str) -> None:
        """Save weight statistics as binary. The ideal extension is `.odds`. """
        path_output_dir = Path(path_output).resolve().parent
        if not os.path.exists(path_output_dir):
            print(f'Output directory is not found. Create: {path_output_dir}')
            os.makedirs(path_output_dir)

        pickle.dump(self.weights_stats, open(path_output, 'wb'))

    def load_weights_stats(self, path_weights: str) -> object:
        """Load trained weight statistics. The default extension is `.odds`."""
        if os.path.isfile(path_weights):
            self.weights_stats = pickle.load(open(path_weights, 'rb'))
        else:
            raise FileExistsError(f'{path_weights} does not exist!')

    def __compute_single_noise_alignment(self, hidden_out_x: Tensor, hidden_out_noise: Tensor,
                                         negative_labels, weight_relevant: Tensor
                                         ) -> Tensor:
        """Compute the odds on noisy examples (1 noise type) for a single input."""
        hidden_output_diff = hidden_out_x - hidden_out_noise
        odds = torch.matmul(hidden_output_diff, weight_relevant.transpose(1, 0))   # Size: [n_samples, n_classes]
        odds = odds[:, negative_labels]  # Size: [n_samples, n_classes-1]
        return odds

    def __compute_multiple_noise_alignments(self, x: Tensor, pred: int,
                                            ) -> OrderedDict:
        """Compute all odds for every noise on a single input."""
        if isinstance(pred, Tensor):
            pred = int(pred.item())

        hidden_out_x = batch_forward(self.latent_net, x.unsqueeze(0),
                                     num_workers=self.num_workers, device=self.device)
        weight_relevant = self.weight_diff[:, pred]
        negative_labels = get_negative_labels(pred, self.n_classes)
        alignments = OrderedDict()
        for noise_eps in self.noise_list:
            x_noisy = create_noisy_examples(x,
                                            n_samples=self.n_noise_samples,
                                            noise_eps=noise_eps,
                                            clip_range=self.noise_clip_range)
            hidden_out_noise = batch_forward(self.latent_net, x_noisy,
                                             num_workers=self.num_workers, device=self.device)
            odds = self.__compute_single_noise_alignment(hidden_out_x,
                                                         hidden_out_noise,
                                                         negative_labels,
                                                         weight_relevant)
            alignments[noise_eps] = odds
        return alignments

    def __collect_weights_stats(self, X: Tensor, preds: Tensor) -> Dict:
        """Collect weights stats from a dataset."""
        weights_stats = {(c, noise_eps): [] for c in range(self.n_classes) for noise_eps in self.noise_list}

        with torch.no_grad():
            pbar = tqdm(zip(X, preds), total=len(X))
            pbar.set_description('Collecting weight stats', refresh=False)
            for _x, _y in pbar:
                label = int(_y.item())
                alignments = self.__compute_multiple_noise_alignments(_x, label)
                for noise_eps in alignments:
                    weights_stats[(label, noise_eps)].append(alignments[noise_eps])

        for key in weights_stats:
            weights = torch.vstack(weights_stats[key])
            # Replace the weights with mean and std.
            weights_stats[key] = {'mean': weights.mean(0), 'std': weights.std(0)}

        return weights_stats


if __name__ == '__main__':
    from baard.classifiers.mnist_cnn import MNIST_CNN
    from baard.utils.torch_utils import dataset2tensor
    from sklearn.model_selection import train_test_split

    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_DATA = os.path.join(PATH_ROOT, 'data')
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf')

    # Parameters for development:
    NOIST_LIST_DEV = ['n0.01', 'u0.01']
    N_NOISE_DEV = 30
    SEED_DEV = 0

    pl.seed_everything(SEED_DEV)

    model = MNIST_CNN.load_from_checkpoint(os.path.join(PATH_CHECKPOINT, 'mnist_cnn.ckpt'))
    detector = OddsAreOddDetector(model,
                                  DATASETS[0],
                                  noise_list=NOIST_LIST_DEV,
                                  n_noise_samples=N_NOISE_DEV)

    PATH_VAL_DATA = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'ValClean.n_1000.pt')
    PATH_WEIGHTS_DEV = os.path.join('temp', 'dev_odds_detector.odds')

    val_dataset = torch.load(PATH_VAL_DATA)
    X_val, y_val = dataset2tensor(val_dataset)
    # Limit the size for quick development. Using stratified sampling to ensure class distribution.
    SIZE_DEV = 100
    _, X_dev, _, y_dev = train_test_split(X_val, y_val, test_size=SIZE_DEV, random_state=SEED_DEV)

    # Train detector
    # detector.train(X_dev, y_dev)
    # detector.save_weights_stats(PATH_WEIGHTS_DEV)

    # Evaluate detector
    detector2 = OddsAreOddDetector(model,
                                   DATASETS[0],
                                   noise_list=NOIST_LIST_DEV,
                                   n_noise_samples=N_NOISE_DEV)
    detector2.load_weights_stats(PATH_WEIGHTS_DEV)
    # for key in detector2.weights_stats:
    #     mean1 = detector.weights_stats[key]['mean']
    #     mean2 = detector2.weights_stats[key]['mean']
    #     assert torch.all(mean1 == mean2)
    #     std1 = detector.weights_stats[key]['std']
    #     std2 = detector2.weights_stats[key]['std']
    #     assert torch.all(std1 == std2)

    scores = detector2.extract_features(X_dev[:30])
    print(scores)

    # Load adversarial examples
    PATH_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'APGD.Linf.n_100.e_0.22.pt')
    adv_dataset = torch.load(PATH_ADV)
    X_adv, y_adv_true = dataset2tensor(adv_dataset)
    scores_adv = detector2.extract_features(X_adv[:30])
    print(scores_adv)
