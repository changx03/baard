"""Implementing the paper "ML-LOO: Detecting Adversarial Examples with Feature
Attribution". -- Yang et. al. (2020)
"""
import logging
import os
import pickle
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.nn import Module, Sequential
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.attacks.apgd import auto_projected_gradient_descent
from baard.classifiers import DATASETS
from baard.detections.base_detector import Detector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.torch_utils import (batch_forward, dataloader2tensor,
                                     get_correct_examples,
                                     get_dataloader_shape,
                                     get_incorrect_examples)

AVAILABLE_STATS_LIST = ('std', 'variance', 'mad', 'kurtosis', 'skewness')
ADV_BATCH_SIZE = 32


logger = logging.getLogger(__name__)


class MLLooDetector(Detector):
    """Implement ML-Loo detector in PyTorch."""

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 device: str = 'cuda',
                 stats_list: Tuple = AVAILABLE_STATS_LIST,
                 attack_eps: float = 0.22,
                 attack_norm: Union[float, int] = np.inf,
                 clip_range: Tuple = (0., 1.),
                 ):
        super().__init__(model, data_name)

        self.stats_list = stats_list

        if not torch.cuda.is_available() and device == 'cuda':
            logger.warning('GPU is not available. Using CPU...')
            device = 'cpu'
        self.device = device

        # Get latent nets based on dataset
        latent_nets, n_classes = self.get_latent_models_and_n_classes(model, data_name)
        self.multi_nets = latent_nets
        self.n_classes = n_classes
        self.attack_eps = attack_eps
        self.attack_norm = attack_norm
        self.clip_range = clip_range

        # Register params
        self.params['device'] = self.device
        self.params['stats_list'] = self.stats_list
        self.params['n_classes'] = self.n_classes
        self.params['attack_eps'] = self.attack_eps
        self.params['attack_norm'] = self.attack_norm
        self.params['clip_range'] = self.clip_range

        # Tunable parameters:
        self.train_mlloss_stats = None
        self.adv_mlloss_stats = None
        self.scaler = None
        self.logistic_regressor = None

    def train(self, X: Tensor, y: Tensor) -> None:
        """Train detector. Train is not required for extracting features.
        """
        if X is None or y is None:
            raise Exception('No sample is passed for training. ML-LOO does not use the entire training set!')

        # Check predictions and true labels
        loader_clean = DataLoader(TensorDataset(X, y),
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=False)
        loader_clean = get_correct_examples(self.model, loader_clean)
        x_corr_shape = get_dataloader_shape(loader_clean)
        if x_corr_shape[0] != len(X):
            logger.warning('%d are classified incorrectly! Use %d examples instead.', len(X) - x_corr_shape.size(0), x_corr_shape)

        logger.info('Train ML-LOO stats on clean training data...')
        X, y = dataloader2tensor(loader_clean)
        # Check if labels match the number of classes
        n_unique = len(y.unique())
        assert n_unique == self.n_classes, f'Number of classes does not match with the labels. Got {n_unique}, expect {self.n_classes}.'
        # Train clean examples
        self.train_mlloss_stats = self.extract_features(X)

        # Train adversarial examples
        X_adv = torch.zeros_like(X)
        dataloader_adv = DataLoader(TensorDataset(X), batch_size=ADV_BATCH_SIZE,
                                    num_workers=os.cpu_count(), shuffle=False)
        start = 0
        pbar = tqdm(dataloader_adv, total=len(dataloader_adv))
        pbar.set_description('Running APGD mini-batch for ML-Loo', refresh=False)
        for batch in pbar:
            x_batch = batch[0]
            end = start + len(x_batch)
            X_adv[start:end] = auto_projected_gradient_descent(
                self.model,
                x_batch,
                norm=self.attack_norm,
                n_classes=self.n_classes,
                eps=self.attack_eps,
                nb_iter=100,
                clip_min=self.clip_range[0],
                clip_max=self.clip_range[1],
            )
            start = end

        logger.info('Train ML-LOO stats on adversarial training data...')
        # Adversarial examples should NOT have same labels as true labels.
        loader_adv = DataLoader(TensorDataset(X_adv, y),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)

        # only uses correctly classified examples.
        loader_adv = get_incorrect_examples(self.model, loader_adv)
        X, y = dataloader2tensor(loader_adv)
        self.adv_mlloss_stats = self.extract_features(X)

        # Train Logistic Regression Model only when adversarial examples are exist.
        n_train = len(self.train_mlloss_stats)
        n_adv = len(self.adv_mlloss_stats)
        X_mlloo = np.vstack([self.train_mlloss_stats, self.adv_mlloss_stats])
        y_mlloo = np.concatenate([np.zeros(n_train), np.ones(n_adv)])

        self.scaler = StandardScaler()
        self.logistic_regressor = LogisticRegressionCV(
            penalty='l1',  # Large feature space, prefer sparse weights
            solver='saga',  # Faster algorithm, but need standardized data.
            max_iter=10000,  # Default param is 100, which does not converge.
            n_jobs=self.num_workers,
        )
        X_mlloo = self.scaler.fit_transform(X_mlloo)
        self.logistic_regressor.fit(X_mlloo, y_mlloo)

    def extract_features(self, X: Tensor, flatten: bool = True) -> ArrayLike:
        """Extract ML-LOO statistics from X."""
        X_mlloo_stats = []
        pbar = tqdm(range(X.size(0)), total=X.size(0))
        pbar.set_description('Compute ML-LOO stats', refresh=False)
        for i in pbar:
            x = X[i]
            one_sample_mlloo_stats = self.__compute_single_mlloo_stats(x)
            X_mlloo_stats.append(one_sample_mlloo_stats)
        X_mlloo_stats = np.array(X_mlloo_stats)

        if flatten:
            n = len(X_mlloo_stats)
            X_mlloo_stats = X_mlloo_stats.reshape(n, -1)

        # Handle nan
        X_mlloo_stats = np.nan_to_num(X_mlloo_stats)
        return X_mlloo_stats

    def predict_proba(self, X: Tensor) -> ArrayLike:
        """Compute probability estimate based on ML-LOO."""
        if self.logistic_regressor is None:
            raise Exception('Logistic regression model is not trained yet!')
        X_mlloo = self.extract_features(X)
        X_mlloo = self.scaler.transform(X_mlloo)
        probs = self.logistic_regressor.predict_proba(X_mlloo)
        return probs[:, 1]  # Only return the 2nd column

    def save(self, path: str = None) -> None:
        """Save trained statistics as binary. The ideal extension is `.mlloo`. """
        if self.train_mlloss_stats is None:
            raise Exception('No trained parameters. Nothing to save.')

        path = create_parent_dir(path, '.mlloo')

        save_obj = {
            'train_mlloss_stats': self.train_mlloss_stats,
            'adv_mlloss_stats': self.adv_mlloss_stats,
            'scaler': self.scaler,
            'logistic_regressor': self.logistic_regressor,
        }
        pickle.dump(save_obj, open(path, 'wb'))

    def load(self, path: str = None) -> None:
        """Load pre-trained statistics. The default extension is `.mlloo`."""
        if os.path.isfile(path):
            save_obj = pickle.load(open(path, 'rb'))
            self.train_mlloss_stats = save_obj['train_mlloss_stats']
            self.adv_mlloss_stats = save_obj['adv_mlloss_stats']
            self.scaler = save_obj['scaler']
            self.logistic_regressor = save_obj['logistic_regressor']
        else:
            raise FileExistsError(f'{path} does not exist!')

    @classmethod
    def get_latent_models_and_n_classes(cls, model: Module, data_name) -> tuple[List, int]:
        """Get latent models and number of classes based on the dataset, e.g., MNIST expects to use a CNN,
        and CIFAR10 expects to use ResNet18.
        """
        models = []
        n_classes = 10
        if data_name == DATASETS[0]:  # MNIST CNN
            models = [
                Sequential(*list(model.children())[:7]),  # Last hidden layer before output (Without ReLU)
                model,    # Model has no SoftMax
            ]
        elif data_name == DATASETS[1]:  # CIFAR10 ResNet18
            resnet18_list = list(model.model.children())
            models = [
                # Sequential(*resnet18_list[:7], nn.AdaptiveMaxPool3d((256, 1, 1))),
                Sequential(*resnet18_list[:-1], nn.AdaptiveMaxPool3d((256, 1, 1))),
                model,
            ]
        else:
            raise NotImplementedError
        return models, n_classes

    @classmethod
    def mean_abs_dev(cls, score: ArrayLike) -> ArrayLike:
        """Mean Absolute Deviation"""
        # score (n, d)
        score = score.reshape(len(score), -1)
        score_mean = np.mean(score, -1, keepdims=True)
        c_score = score - score_mean
        c_score = np.abs(c_score)
        return np.mean(c_score, axis=-1)

    @classmethod
    def med_abs_dev(cls, score: ArrayLike) -> ArrayLike:
        """Median Absolute Deviation"""
        pd = []
        for i in range(len(score)):
            d = score[i]
            median = np.median(d)
            abs_dev = np.abs(d - median)
            _med_abs_dev = np.median(abs_dev)
            pd.append(_med_abs_dev)
        pd = np.array(pd)
        return pd

    @classmethod
    def med_pdist(cls, score: ArrayLike) -> ArrayLike:
        """Median Pairwise Distance"""
        pd = []
        for i in range(len(score)):
            d = score[i]
            k = np.median(pdist(d.reshape(-1, 1)))
            pd.append(k)
        pd = np.array(pd)
        return pd

    @classmethod
    def mean_pdist(cls, score: ArrayLike) -> ArrayLike:
        """Mean Pairwise Distance"""
        pd = []
        for i in range(len(score)):
            d = score[i]
            k = np.mean(pdist(d.reshape(-1, 1)))
            pd.append(k)
        pd = np.array(pd)
        return pd

    @classmethod
    def neg_kurtosis(cls, score: ArrayLike) -> ArrayLike:
        """Negative Kurtosis"""
        k = []
        for i in range(len(score)):
            di = score[i]
            ki = kurtosis(di, nan_policy='omit')
            k.append(ki)
        k = np.array(k)
        return -k

    @classmethod
    def quantile(cls, score: ArrayLike) -> ArrayLike:
        """Between 25-75 Quantile"""
        # score (n, d)
        score = score.reshape(len(score), -1)
        score_75 = np.percentile(score, 75, -1)
        score_25 = np.percentile(score, 25, -1)
        score_qt = score_75 - score_25
        return score_qt

    @classmethod
    def calculate_stats(cls, net_outputs: ArrayLike, stats_name: str) -> ArrayLike:
        """Compute statistics metrics."""
        # net_outputs.shape == (N_SAMPLES, N_LATENT_OUTPUTS)
        # We are looking at the features vertically, so results should be in (N_LATENT_OUTPUTS,).
        if stats_name == 'std':
            results = np.std(net_outputs, axis=-1)
        elif stats_name == 'variance':
            results = np.var(net_outputs, axis=-1)
        elif stats_name == 'con':
            results = cls.mean_abs_dev(net_outputs)
        elif stats_name == 'mad':
            results = cls.med_abs_dev(net_outputs)
        elif stats_name == 'pdist':
            results = cls.mean_pdist(net_outputs)
        elif stats_name == 'med_pdist':
            results = cls.med_pdist(net_outputs)
        elif stats_name == 'kurtosis':
            results = cls.neg_kurtosis(net_outputs)
        elif stats_name == 'skewness':
            results = -skew(net_outputs, axis=-1, nan_policy='omit')
        elif stats_name == 'quantile':
            results = cls.quantile(net_outputs)
        return results

    def __compute_single_mlloo_stats(self, x: Tensor) -> ArrayLike:
        """Compute ML-LOO statistics for a single example."""
        one_sample_mlloo_maps = self.__compute_single_mlloo_maps(x)
        one_sample_mlloo_stats = self.__compute_single_mlloo_stats_from_maps(one_sample_mlloo_maps)
        return one_sample_mlloo_stats

    def __compute_single_mlloo_maps(self, x: Tensor) -> Tensor:
        """Compute ML-LOO maps for one example"""
        x_loo = self.__create_loo_input(x)
        x_mlloo_maps = []
        for net in self.multi_nets:
            loo_outputs = batch_forward(
                model=net,
                X=x_loo,
                batch_size=self.batch_size,
                device=self.device,
                num_workers=self.num_workers,
            )
            # LOO output - original output (at last example)
            loo_outputs_1layer = loo_outputs[:-1] - loo_outputs[-1]
            loo_outputs_1layer = torch.flatten(loo_outputs_1layer, start_dim=1)
            x_mlloo_maps.append(loo_outputs_1layer)
        x_mlloo_maps = torch.hstack(x_mlloo_maps).transpose(0, 1)
        # Each LOO map has the same size of input features, and there're n_hidden_neurons of them.
        return x_mlloo_maps  # In shape of (n_hidden_neurons, n_input_features).

    def __compute_single_mlloo_stats_from_maps(self, mlloo_maps: Tensor):
        """Compute statistics based on ML-LOO maps of a single example."""
        mlloo_maps = mlloo_maps.detach().numpy()  # Work with Numpy now.
        mlloo_stats = []
        for stats_name in self.stats_list:
            single_stats = self.calculate_stats(mlloo_maps, stats_name)
            mlloo_stats.append(single_stats)
        mlloo_stats = np.vstack(mlloo_stats)
        return mlloo_stats

    def __create_loo_input(self, x: Tensor) -> Tensor:
        """Create a LOO mapping for 1 example. Output is in shape of (n_features + 1, n_features).
        It includes masking 1 input with 0 for every feature + the original input.
        NOTE: Expecting only one example, not a list!
        """
        # Get the actual number of input features
        n_features = int(torch.Tensor(list(x.size())).prod().item())
        # The output should be (n_features + 1, <ORIGINAL_SAMPLE_SHAPE>)
        shape_repeat = [n_features + 1] + [1 for _ in range(len(x.size()))]

        # Expand the dimension. Expecting only one example!
        X_repeated = x.unsqueeze(0).repeat(shape_repeat)

        # Create a mask with 0 for each pixel. Keep the last example original.
        X_loo_mask = torch.ones_like(X_repeated)
        loo_shape = X_loo_mask.size()
        X_loo_mask = X_loo_mask.reshape(len(X_loo_mask), -1)
        for i in range(X_loo_mask.size(-1)):
            X_loo_mask[i, i] = 0
        X_loo_mask = X_loo_mask.reshape(loo_shape)

        X_loo = X_loo_mask * X_repeated
        return X_loo
