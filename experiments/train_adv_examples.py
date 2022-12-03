"""Train adversarial examples"""
import json
import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.attacks import ATTACKS
from baard.attacks.apgd import auto_projected_gradient_descent
from baard.attacks.cw2 import carlini_wagner_l2
from baard.attacks.fast_gradient_method import fast_gradient_method
from baard.attacks.projected_gradient_descent import projected_gradient_descent
from baard.classifiers import DATASETS
from baard.utils.miscellaneous import filter_exist_eps, norm_parser
from baard.utils.torch_utils import (dataloader2tensor, dataset2tensor,
                                     get_correct_examples)

PATH_ROOT = os.getcwd()
PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf')
ADV_BATCH_SIZE = 32  # Training adversarial examples in small batches.


def check_file_exist(path_file):
    """Shortcut for check a file is exist."""
    if not os.path.isfile(path_file):
        raise FileExistsError(f"{path_file} does not exist!")
    return True


def get_model(data: str) -> LightningModule:
    """Return a PyTorch Lightning Module."""
    from baard.classifiers.cifar10_resnet18 import CIFAR10_ResNet18
    from baard.classifiers.mnist_cnn import MNIST_CNN

    model = None
    if data == DATASETS[0]:
        path_checkpoint = os.path.join(PATH_CHECKPOINT, 'mnist_cnn.ckpt')
        check_file_exist(path_checkpoint)
        model = MNIST_CNN.load_from_checkpoint(path_checkpoint)
    elif data == DATASETS[1]:
        path_checkpoint = os.path.join(PATH_CHECKPOINT, 'cifar10_resnet18.ckpt')
        check_file_exist(path_checkpoint)
        model = CIFAR10_ResNet18.load_from_checkpoint(path_checkpoint)
    else:
        raise NotImplementedError()
    return model


def get_attack(attack_name: str, adv_params: dict):
    """Return an Adversarial Attack function."""
    adv_params = dict(adv_params)  # Create a copy
    attack = None
    if attack_name == ATTACKS[0]:
        attack = fast_gradient_method
    elif attack_name == ATTACKS[1]:
        attack = projected_gradient_descent
    elif attack_name == ATTACKS[2]:
        attack = carlini_wagner_l2
        if 'eps' in adv_params.keys():
            del adv_params['eps']
        if 'eps_iter' in adv_params.keys():
            del adv_params['eps_iter']
        if 'nb_iter' in adv_params.keys():
            del adv_params['nb_iter']
    elif attack_name == ATTACKS[3]:
        attack = auto_projected_gradient_descent
    else:
        raise NotImplementedError()
    return attack, adv_params


def generate_adv_examples(data: str,
                          attack_name: str,
                          eps: ArrayLike,
                          adv_params: dict,
                          path_outputs: str,
                          n_att: int,
                          n_val: int,
                          seed: int):
    """Generate adversarial examples based on given epsilons (perturbation)."""
    pl.seed_everything(seed)

    # Step 1: Get correct labelled data
    model = get_model(data)
    val_loader = model.val_dataloader()
    num_workers = os.cpu_count()
    print('num_workers:', num_workers)
    path_correct_val_dataset = os.path.join(path_outputs, 'CorrectValDataset.pt')

    if not os.path.isfile(path_correct_val_dataset):
        dataset_val_correct = get_correct_examples(model, val_loader, return_loader=False)
        print(f'{len(dataset_val_correct)} examples are correctly classified.')
        torch.save(dataset_val_correct, path_correct_val_dataset)
    else:
        dataset_val_correct = torch.load(path_correct_val_dataset)
        print('Load existing `CorrectValDataset.pt`...')
    loader_val_correct = DataLoader(dataset_val_correct, batch_size=val_loader.batch_size,
                                    num_workers=num_workers, shuffle=False)
    _dataset = get_correct_examples(model, loader_val_correct, return_loader=False)
    print(f'{len(_dataset) / len(dataset_val_correct) * 100}% out of {len(dataset_val_correct)} examples are correctly classified.')
    del _dataset

    # Step 2: Split the data
    if n_att + n_val > len(dataset_val_correct):
        raise ValueError('The total number of adversarial and validation examples are larger than the test set.')

    X_val, y_val = dataloader2tensor(loader_val_correct)
    path_adv_clean = os.path.join(path_outputs, f'AdvClean-{n_att}.pt')
    if not os.path.isfile(path_adv_clean):
        # `train_test_split` can work directly with PyTorch Tensor
        X_leftover, X_adv_clean, y_leftover, y_adv_clean = train_test_split(X_val, y_val, test_size=n_att, random_state=seed)
        assert len(X_adv_clean) == n_att
        torch.save(TensorDataset(X_adv_clean, y_adv_clean), path_adv_clean)

        # Validation set can only create from initial split!
        if n_val > 0:
            _, X_val, _, y_val = train_test_split(X_leftover, y_leftover, test_size=n_val, random_state=seed)
            assert len(X_val) == n_val
            path_val_clean = os.path.join(path_outputs, f'ValClean-{n_val}.pt')
            torch.save(TensorDataset(X_val, y_val), path_val_clean)
    else:
        print(f'Load from {path_adv_clean}')
        dataset_adv_clean = torch.load(path_adv_clean)
        dataloader_adv_clean = DataLoader(dataset_adv_clean, batch_size=val_loader.batch_size,
                                          num_workers=num_workers, shuffle=False)
        X_adv_clean, y_adv_clean = dataloader2tensor(dataloader_adv_clean)

        path_val_clean = os.path.join(path_outputs, f'ValClean-{n_val}.pt')
        if n_val > 0 and not os.path.isfile(path_val_clean):
            print(f'WARNING: Validation dataset is missing! Delete `AdvClean-{n_att}.pt` and run the code again!')
    # del X_val, y_val, dataset_val_correct, loader_val_correct

    # Step 3: Generate adversarial examples
    # Same trainer, the model has no change.
    trainer = pl.Trainer(accelerator='auto',
                         logger=False,
                         enable_model_summary=False,
                         enable_progress_bar=False)
    attack, adv_params = get_attack(attack_name, adv_params)

    # NOTE: C&W is only on L2 for now.
    attack_norm = 2 if attack_name == ATTACKS[2] else str(adv_params['norm'])
    path_log_results = os.path.join(path_outputs, f'{attack_name}-L{attack_norm}-SuccessRate.csv')

    # # Get epsilon which hasn't rained.
    # eps = filter_exist_eps(eps,
    #                        path_outputs,
    #                        attack_name,
    #                        lnorm=norm_parser(adv_params['norm']),
    #                        n=X_adv_clean.size(0))
    # if len(eps) == 0:
    #     print('No epsilon need to train. Exit.')
    #     return

    with open(path_log_results, 'a', encoding='UTF-8') as file:
        file.write(','.join(['eps', 'success_rate']) + '\n')
        for e in eps:
            try:
                adv_params['eps'] = e
                # Epsilon represent param `confidence` in C&W attack.
                if attack_name == ATTACKS[2]:
                    del adv_params['eps']
                    adv_params['confidence'] = e

                path_adv = os.path.join(path_outputs, f'{attack_name}-L{attack_norm}-{n_att}-{e}.pt')
                # Check results
                if os.path.exists(path_adv):
                    print(f'Found {path_adv} Skip!')
                    dataset_adv = torch.load(path_adv)
                    X_adv, _ = dataset2tensor(dataset_adv)
                else:
                    print('Training advx on test set...')
                    X_adv = torch.zeros_like(X_adv_clean)
                    loader_val_correct = DataLoader(TensorDataset(X_adv_clean), batch_size=ADV_BATCH_SIZE,
                                                    num_workers=num_workers, shuffle=False)
                    start = 0
                    pbar = tqdm(loader_val_correct, total=len(loader_val_correct))
                    pbar.set_description(f'Running {attack_name} eps/c={e} attack')
                    for batch in pbar:
                        x_batch = batch[0]
                        end = start + len(x_batch)
                        X_adv[start:end] = attack(model, x_batch, **adv_params)
                        start = end

                    # Save adversarial examples
                    print(f'Save to {path_adv}')
                    torch.save(TensorDataset(X_adv, y_adv_clean), path_adv)

                # Checking results
                dataset_adv = TensorDataset(X_adv)
                loader_adv = DataLoader(dataset_adv, batch_size=val_loader.batch_size, num_workers=num_workers, shuffle=False)
                outputs_adv = torch.vstack(trainer.predict(model, loader_adv))
                preds_adv = torch.argmax(outputs_adv, dim=1)
                success_rate = (preds_adv == y_adv_clean).float().mean() * 100

                print(f'[e={e}]{success_rate}% out of {len(preds_adv)} examples are correctly classified.')
                file.write(','.join([f'{i}' for i in [e, success_rate]]) + '\n')

                # Also train adversarial examples on ``ValClean''
                path_adv_val = os.path.join(path_outputs, f'{attack_name}-L{attack_norm}-{n_att}-{e}-val.pt')
                # Check results
                if os.path.exists(path_adv_val):
                    print(f'Found {path_adv_val} Skip!')
                else:
                    print('Training advx on validation set...')
                    path_val_clean = os.path.join(path_outputs, f'ValClean-{n_val}.pt')
                    dataset_val = torch.load(path_val_clean)
                    X_val, y_val = dataset2tensor(dataset_val)
                    X_adv_val = torch.zeros_like(X_val)
                    loader_val_correct = DataLoader(TensorDataset(X_val), batch_size=ADV_BATCH_SIZE,
                                                    num_workers=num_workers, shuffle=False)
                    start = 0
                    pbar = tqdm(loader_val_correct, total=len(loader_val_correct))
                    pbar.set_description(f'Running {attack_name} eps/c={e} attack')
                    for batch in pbar:
                        x_batch = batch[0]
                        end = start + len(x_batch)
                        X_adv_val[start:end] = attack(model, x_batch, **adv_params)
                        start = end

                    # Save adversarial examples
                    print(f'Save to {path_adv_val}')
                    torch.save(TensorDataset(X_adv_val, y_val), path_adv_val)
            except BaseException as err:
                print(f'WARNING: Catch an exception: {err}')


def parse_arguments():
    """Parse command line arguments.
    Examples:
    For quick develop only. Set `n_att` to a larger value when running the experiment!
    Data: MNIST, Attack: FGSM
    python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":"inf"}' --eps="[0.22,0.66]" --n_val=1000
    python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":2}' --eps="[1,4,8]" --n_val=1000

    Data: MNIST, Attack: PGD
    python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.22,0.66]" --n_val=1000
    python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":2, "eps_iter":0.1}' --eps="[1,4,8]" --n_val=1000

    Data: MNIST, Attack: APGD
    python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.22,0.66]" --n_val=1000
    python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":2, "eps_iter":0.1}' --eps="[1,4,8]" --n_val=1000

    Data: MNIST, Attack: CW2
    python ./experiments/train_adv_examples.py -d=MNIST --attack=CW2 --params='{"max_iterations": 200}' --eps="[0]" --n_val=1000

    Data: CIFAR10, Attack: FGSM
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":"inf"}' --eps="[0.01,0.1,0.3]" --n_val=1000
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":2}' --eps="[0.3,3]" --n_val=1000

    Data: CIFAR10, Attack: PGD
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.01,0.1,0.3]" --n_val=1000
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":2, "eps_iter":0.1}' --eps="[0.3,3]" --n_val=1000

    Data: CIFAR10, Attack: APGD
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.01,0.1,0.3]" --n_val=1000
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":2, "eps_iter":0.1}' --eps="[0.3,3]" --n_val=1000

    Data: CIFAR10, Attack: CW2
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=CW2 --params='{"max_iterations": 200}' --eps="[0]" --n_val=1000
    """
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('-d', '--data', default=DATASETS[0], choices=DATASETS)
    parser.add_argument('-o', '--output', type=str, default='results')
    # NOTE: Default value is for quick developing only. Use 1000 for actual experiments.
    parser.add_argument('--n_att', type=int, default=100)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('-a', '--attack', default='APGD', choices=ATTACKS)
    parser.add_argument('--eps', type=json.loads, default='[1,4,8]', required=True,
                        help='A list of epsilons as a JSON string. e.g., "[0.06, 0.13, 0.25]".')
    parser.add_argument('--params', type=json.loads, default='{"norm":2, "eps_iter":0.1}',
                        help='Parameters for the adversarial attack as a JSON string. e.g., \'{"norm":"inf"}\'.')
    args = parser.parse_args()
    seed = args.seed
    data = args.data
    n_att = args.n_att
    n_val = args.n_val
    attack_name = args.attack
    eps_list = args.eps
    adv_params = args.params if args.params is not None else dict()

    eps_list = np.round(eps_list, 2).astype(float)  # Use float numbers.

    print('PATH_ROOT', PATH_ROOT)
    print('SEED:', seed)
    print('DATA:', data)
    print('N_ATT:', n_att)
    print('N_VAL:', n_val)
    print('ATTACK:', attack_name)
    print('EPS:', eps_list)

    if 'norm' in adv_params.keys() and adv_params['norm'] == 'inf':
        adv_params['norm'] = np.inf
    else:
        adv_params['norm'] = 2
    print('PARAMS:', adv_params)

    path_outputs = os.path.join(PATH_ROOT, args.output, f'exp{seed}', data)
    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)
        print(f'Creates dir: {path_outputs}')

    return data, attack_name, eps_list, adv_params, path_outputs, n_att, n_val, seed


def main():
    """Main pipeline for generating adversarial examples."""
    data, attack_name, eps, adv_params, path_outputs, n_att, n_val, seed = parse_arguments()
    generate_adv_examples(data, attack_name, eps, adv_params, path_outputs, n_att, n_val, seed)


if __name__ == '__main__':
    main()
