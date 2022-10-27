"""Train adversarial examples"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to system path
sys.path.append(os.getcwd())
print(sys.path)

PATH_ROOT = os.getcwd()
PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf')
DATASETS = ['MNIST', 'CIFAR10']
ATTACKS = ['FGSM', 'PGD', 'CW2', 'APGD']
ADV_BATCH_SIZE = 32  # Training adversarial examples in small batches.


def check_file_exist(path_file):
    if not os.path.isfile(path_file):
        raise FileExistsError(f"{path_file} does not exist!")
    return True


def get_model(data: str) -> LightningModule:
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
    from baard.attacks.apgd import auto_projected_gradient_descent
    from baard.attacks.cw2 import carlini_wagner_l2
    from baard.attacks.fast_gradient_method import fast_gradient_method
    from baard.attacks.projected_gradient_descent import projected_gradient_descent

    adv_params = dict(adv_params)  # Create a copy
    attack = None
    if attack_name == ATTACKS[0]:
        attack = fast_gradient_method
    elif attack_name == ATTACKS[1]:
        attack = projected_gradient_descent
    elif attack_name == ATTACKS[2]:
        attack = carlini_wagner_l2
        if 'norm' in adv_params.keys():
            del adv_params['norm']
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


def generate_adv_examples(
    data: str,
    attack_name: str,
    eps: list,
    adv_params: dict,
    path_outputs: str,
    n_att: int,
    n_val: int,
    seed: int
):
    """Generate adversarial examples based on given epsilons (perturbation)."""
    from baard.utils.torch_utils import dataloader2tensor, get_correct_examples

    # Step 1: Get correct labelled data
    model = get_model(data)
    val_loader = model.val_dataloader()
    path_correct_val_dataset = os.path.join(path_outputs, 'CorrectValDataset.pt')
    if not os.path.isfile(path_correct_val_dataset):
        dataset = get_correct_examples(model, val_loader, return_loader=False)
        print(f'{len(dataset)} examples are correctly classified.')
        torch.save(dataset, path_correct_val_dataset)
    else:
        dataset = torch.load(path_correct_val_dataset)
        print(f'Load existing `CorrectValDataset.pt`...')
    dataloader = DataLoader(dataset, batch_size=val_loader.batch_size,
                            num_workers=val_loader.num_workers, shuffle=False)
    _dataset = get_correct_examples(model, dataloader, return_loader=False)
    print(f'{len(_dataset) / len(dataset) * 100}% out of {len(dataset)} examples are correctly classified.')
    del _dataset

    # Step 2: Split the data
    if n_att + n_val > len(dataset):
        raise ValueError('The total number of adversarial and validation examples are larger than the test set.')

    x, y = dataloader2tensor(dataloader)
    path_adv_clean = os.path.join(path_outputs, f'AdvClean.n_{n_att}.pt')
    if not os.path.isfile(path_adv_clean):
        # `train_test_split` can work directly with PyTorch Tensor
        X_leftover, X_adv_clean, y_leftover, y_adv_clean = train_test_split(x, y, test_size=n_att, random_state=seed)
        assert len(X_adv_clean) == n_att
        torch.save(TensorDataset(X_adv_clean, y_adv_clean), path_adv_clean)

        # Validation set can only create from initial split!
        if n_val > 0:
            _, X_val, _, y_val = train_test_split(X_leftover, y_leftover, test_size=n_val, random_state=seed)
            assert len(X_val) == n_val
            path_val_clean = os.path.join(path_outputs, f'ValClean.n_{n_val}.pt')
            torch.save(TensorDataset(X_val, y_val), path_val_clean)
    else:
        print(f'Load existing `ValClean.n_{n_val}.pt`...')
        dataset_adv_clean = torch.load(path_adv_clean)
        dataloader_adv_clean = DataLoader(dataset_adv_clean, batch_size=val_loader.batch_size,
                                          num_workers=val_loader.num_workers, shuffle=False)
        X_adv_clean, y_adv_clean = dataloader2tensor(dataloader_adv_clean)

        path_val_clean = os.path.join(path_outputs, f'ValClean.n_{n_val}.pt')
        if n_val > 0 and not os.path.isfile(path_val_clean):
            print(f'WARNING: Validation dataset is missing! Delete `AdvClean.n_{n_att}.pt` and run the code again!')
    del x, y, dataset, dataloader

    # Step 3: Generate adversarial examples
    # Same trainer, the model has no change.
    trainer = pl.Trainer(accelerator='auto', logger=False)
    attack, adv_params = get_attack(attack_name, adv_params)

    # NOTE: C&W is only on L2 for now.
    attack_norm = 2 if attack_name == ATTACKS[2] else str(adv_params['norm'])
    path_log_results = os.path.join(path_outputs, f'{attack_name}_L{attack_norm}_success_rate.csv')

    with open(path_log_results, 'a') as file:
        file.write(','.join(['eps', 'success_rate']) + '\n')
        for e in eps:
            try:
                adv_params['eps'] = e
                # Epsilon represent param `confidence` in C&W attack.
                if attack_name == ATTACKS[2]:
                    del adv_params['eps']
                    adv_params['confidence'] = e

                X_adv = torch.zeros_like(X_adv_clean)
                dataloader = DataLoader(TensorDataset(X_adv_clean), batch_size=ADV_BATCH_SIZE,
                                        num_workers=val_loader.num_workers, shuffle=False)
                start = 0
                pbar = tqdm(enumerate(dataloader), total=len(dataloader))
                pbar.set_description(f'Running {attack_name} eps/c={e} attack')
                for i, batch in pbar:
                    x = batch[0]
                    end = start + len(x)
                    X_adv[start:end] = attack(model, x, **adv_params)
                    start = end

                # Save adversarial examples
                path_adv = os.path.join(path_outputs, f'{attack_name}.L{attack_norm}.n_{n_att}.e_{e}.pt')
                torch.save(TensorDataset(X_adv, y_adv_clean), path_adv)

                # Checking results
                dataset_adv = TensorDataset(X_adv)
                loader_adv = DataLoader(dataset_adv, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False)
                outputs_adv = torch.vstack(trainer.predict(model, loader_adv))
                preds_adv = torch.argmax(outputs_adv, dim=1)
                success_rate = (preds_adv == y_adv_clean).float().mean() * 100

                print(f'[e={e}]{success_rate}% out of {len(preds_adv)} examples are correctly classified.')
                file.write(','.join([f'{i}' for i in [e, success_rate]]) + '\n')
            except BaseException as err:
                print(f'WARNING: Catch an exception: {err}')


if __name__ == '__main__':
    """Examples:
    # For quick develop only. Set `n_att` to a larger value when running the experiment!
    # Data: MNIST, Attack: FGSM
    python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":"inf"}' --eps="[0.03,0.12,0.31]"
    python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":2}' --eps="[1, 2, 4]"

    # Data: MNIST, Attack: PGD
    python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.31]"
    python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":2, "eps_iter":0.1}' --eps="[1, 4]"

    # Data: MNIST, Attack: APGD
    python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.31]"
    python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":2, "eps_iter":0.1}' --eps="[1, 4]"

    # Data: MNIST, Attack: CW2
    python ./experiments/train_adv_examples.py -d=MNIST --attack=CW2 --params='{"max_iterations": 200}' --eps="[0]"

    # Data: CIFAR10, Attack: FGSM
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":"inf"}' --eps="[0.03,0.09,0.16]"
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":2}' --eps="[0.3,2]"

    # Data: CIFAR10, Attack: PGD
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.16]"
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":2, "eps_iter":0.1}' --eps="[0.3,2]"

    # Data: CIFAR10, Attack: APGD
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.16]"
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":2, "eps_iter":0.1}' --eps="[0.3,2]"

    # # Data: CIFAR10, Attack: CW2
    python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=CW2 --params='{"max_iterations": 200}' --eps="[0]"
    """
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('-d', '--data', default=DATASETS[0], choices=DATASETS)
    parser.add_argument('-o', '--output', type=str, default='results')
    # NOTE: Default value is for quick developing only. Use 1000 for actual experiments.
    parser.add_argument('--n_att', type=int, default=100)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('-a', '--attack', default=ATTACKS[0], choices=ATTACKS)
    parser.add_argument('--eps', type=json.loads, default='[0.06]',
                        help='A list of epsilons as a JSON string. e.g., "[0.06, 0.13, 0.25]".')
    parser.add_argument('--params', type=json.loads, default='{"norm":"inf"}',
                        help='Parameters for the adversarial attack as a JSON string. e.g., \'{"norm":"inf"}\'.')
    args = parser.parse_args()
    seed = args.seed
    data = args.data
    n_att = args.n_att
    n_val = args.n_val
    attack = args.attack
    eps = args.eps
    adv_params = args.params if args.params is not None else dict()

    print('PATH_ROOT', PATH_ROOT)
    print('SEED:', seed)
    print('DATA:', data)
    print('N_ATT:', n_att)
    print('N_VAL:', n_val)
    print('ATTACK:', attack)
    print('EPS:', eps)

    if 'norm' in adv_params.keys() and adv_params['norm'] == 'inf':
        adv_params['norm'] = np.inf
    print('PARAMS:', adv_params)

    path_outputs = os.path.join(PATH_ROOT, args.output, f'exp{seed}', data)
    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)
        print(f'Creates dir: {path_outputs}')

    pl.seed_everything(seed)

    generate_adv_examples(data, attack, eps, adv_params, path_outputs, n_att, n_val, seed)
