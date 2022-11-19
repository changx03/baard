"""Using the targeted white-box PGD attack to generate adversarial examples."""
import logging
import os
from argparse import ArgumentParser
from typing import Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import baard.classifiers as clf
import baard.utils.torch_utils as tr_utils
from baard.attacks.adaptive_whitebox import targeted_whitebox_pgd
from baard.detections.baard_applicability import ApplicabilityStage

logging.basicConfig(level=logging.INFO)


def get_model(data_name: str, path_checkpoint: str) -> LightningModule:
    """Get the pre-trained classifier."""
    if data_name == 'MNIST':
        checkpoint_name = 'mnist_cnn.ckpt'
        model = clf.MNIST_CNN.load_from_checkpoint(os.path.join(path_checkpoint, checkpoint_name))
    elif data_name == 'CIFAR10':
        checkpoint_name = 'cifar10_resnet18.ckpt'
        model = clf.CIFAR10_ResNet18.load_from_checkpoint(os.path.join(path_checkpoint, checkpoint_name))
    else:
        raise NotImplementedError
    return model


def load_data(seed: int, data_name: str, path: str = 'results',
              filename_clean: str = 'AdvClean-1000.pt',
              filename_val: str = 'ValClean-1000.pt'
              ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load clean data"""
    path_base = os.path.join(path, f'exp{seed}', data_name)
    dataset_clean = torch.load(os.path.join(path_base, filename_clean))
    X_clean, y_clean = tr_utils.dataset2tensor(dataset_clean)
    dataset_val = torch.load(os.path.join(path_base, filename_val))
    X_val, y_val = tr_utils.dataset2tensor(dataset_val)
    return X_val, y_val, X_clean, y_clean


def find_targets(model: LightningModule, X: Tensor, X_val: Tensor,
                 latent_net: torch.nn.Sequential = None, num_workers: int = None
                 ) -> Tensor:
    """Find target examples for each X."""
    # Find the original prediction from the classifier. Avoid label leaking.
    if num_workers is None:
        num_workers = os.cpu_count()
    dataloader = DataLoader(
        TensorDataset(X), batch_size=128, shuffle=False, num_workers=num_workers)
    Y = tr_utils.predict(model, dataloader)

    dataloader = DataLoader(
        TensorDataset(X_val), batch_size=128, shuffle=False, num_workers=num_workers)
    Y_val = tr_utils.predict(model, dataloader)

    if latent_net is None:  # the distance is computed on inputs.
        feature_val = X_val.reshape(len(X_val), -1)
        feature = X.reshape(len(X), -1)
    else:
        feature_val = tr_utils.batch_forward(latent_net, X_val)
        feature_val = feature_val.reshape(len(feature_val), -1)

        feature = tr_utils.batch_forward(latent_net, X)
        feature = feature.reshape(len(feature), -1)

    X_nearest = torch.zeros_like(X)
    for i in range(X.size(0)):
        one_feature = feature[i]
        y = Y[i]

        # Find a subset that has different label than y
        indices_target = torch.where(Y_val != y)[0]
        feature_target = feature_val[indices_target]
        X_target = X_val[indices_target]
        Y_target = Y_val[indices_target]

        cosine_sim_fn = torch.nn.CosineSimilarity(dim=1)
        cos_sim_target = cosine_sim_fn(
            feature_target,
            one_feature,
        )
        angular_dist_target = torch.arccos(cos_sim_target) / torch.pi
        idx_nearest = torch.argmin(angular_dist_target)
        x_target = X_target[idx_nearest]
        y_target = Y_target[idx_nearest]
        assert y_target != y
        X_nearest[i] = x_target
    return X_nearest


def generate_adv(model: LightningModule, X_clean: Tensor, X_target: Tensor,
                 eps_iter: float, norm: Union[float, int], eps: float,
                 n_iter: int = 100, batch_size: int = 32, num_workers: int = None
                 ) -> Tensor:
    """Generate targeted white-box adaptive adversarial examples"""
    if num_workers is None:
        num_workers = os.cpu_count()
    dataset = TensorDataset(X_clean, X_target)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    start = 0
    X_adv = torch.zeros_like(X_clean)
    pbar = tqdm(dataloader, total=len(dataloader))
    pbar.set_description('Running whitebox attack', refresh=False)
    for batch in pbar:
        x, x_tar = batch
        x_adv = targeted_whitebox_pgd(model.to('cpu'), x, x_tar,
                                      eps=eps,
                                      eps_iter=eps_iter,
                                      nb_iter=n_iter,
                                      norm=norm,
                                      rand_init=False,
                                      c=1.,
                                      early_stop=False,
                                      )
        end = start + x.size(0)
        X_adv[start: end] = x_adv
        start = end
    return X_adv


def parse_arguments():
    """Parse command line arguments.
    Example:
    python ./experiments/baard_whitebox.py --data MNIST --seed 1234 --clean "AdvClean-100.pt" --norm 2 --epsiter 0.1 --eps 4
    python ./experiments/baard_whitebox.py --data MNIST --seed 1234 --clean "AdvClean-100.pt" --norm inf --epsiter 0.03 --eps 0.22
    python ./experiments/baard_whitebox.py --data CIFAR10 --seed 1234 --clean "AdvClean-100.pt" --norm 2 --epsiter 0.05 --eps 0.3
    python ./experiments/baard_whitebox.py --data CIFAR10 --seed 1234 --clean "AdvClean-100.pt" --norm inf --epsiter 0.01 --eps 0.01
    """
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, choices=clf.DATASETS, required=True)
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('--clean', type=str, default='AdvClean-1000.pt')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--norm', type=str, choices=['inf', '2'], default=2)
    parser.add_argument('--epsiter', type=float, default=0.1)
    parser.add_argument('--eps', type=float, default=4.)
    parser.add_argument('--val', type=str, default='ValClean-1000.pt')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    data_name = args.data
    seed = args.seed
    path_output = args.output
    if path_output is None:
        path_output = os.path.join('results', f'exp{seed}', data_name)
    norm = args.norm
    if norm == 'inf':
        norm = np.inf
    elif norm == '2':
        norm = 2
    else:
        raise ValueError(f'Norm must be either inf or 2. Got {norm}!')
    eps_iter = args.epsiter
    eps = args.eps
    filename_clean = args.clean
    filename_val = args.val
    path_checkpoint = args.checkpoint
    if path_checkpoint is None:
        path_checkpoint = 'pretrained_clf'

    return {
        'data_name': data_name,
        'seed': seed,
        'path_output': path_output,
        'norm': norm,
        'eps_iter': eps_iter,
        'eps': eps,
        'filename_clean': filename_clean,
        'filename_val': filename_val,
        'path_checkpoint': path_checkpoint,
    }


def main():
    """The main pipeline."""
    args = parse_arguments()
    print(args)

    data_name = args['data_name']
    seed = args['seed']
    path_output = args['path_output']
    norm = args['norm']
    eps_iter = args['eps_iter']
    eps = args['eps']
    filename_clean = args['filename_clean']
    filename_val = args['filename_val']
    path_checkpoint = args['path_checkpoint']

    pl.seed_everything(seed)

    model = get_model(data_name, path_checkpoint)
    X_val, y_val, X_clean, y_clean = load_data(
        seed, data_name,
        filename_clean=filename_clean, filename_val=filename_val)
    latent_net = ApplicabilityStage.get_latent_net(model, data_name=data_name)
    X_nearest_target = find_targets(model, X_clean, X_val, latent_net=latent_net)
    X_adv = generate_adv(model, X_clean, X_nearest_target,
                         eps_iter=eps_iter, norm=norm, eps=eps,
                         n_iter=100, batch_size=32)

    path_output = os.path.join(path_output, f'whitebox-L{norm}-{X_adv.size(0)}-{eps}.pt')
    print('Save results to:', path_output)
    torch.save(TensorDataset(X_adv, y_clean), path_output)


if __name__ == '__main__':
    main()