import random
import time
from pathlib import Path

import torch
import upsilonconf
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
from upsilonconf import save_config

from mlp_search.search import get_model
from pre_processing import Whiten
from trainer import Trainer
from utils import make_deterministic


def get_data(name: str, root: str, pre_process: str = "normal"):
    """
    Create train- and valid dataset for experiment.

    Parameters
    ----------
    name : str
        The name of the dataset to use.
    root : str
        Path to where the data is stored on disk.
    train_split : float, optional
        The fraction of data to use for training.

    Returns
    -------
    (train, valid) : tuple of two Datasets
        A tuple of datasets for updating and evaluating the network.
    (in_shape, class_count) : tuple of torch.Size and int
        A tuple with the input shape and number of classes for this dataset.
    """
    name = name.lower().strip()
    pre_process = pre_process.lower().strip()

    if name == "mnist":
        dataset = datasets.MNIST
        mean, std = (0.1307, ), (0.3081, )
        shapes = torch.Size((1, 28, 28)), 10
    elif name == "cifar10":
        dataset = datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, .2435, 0.2616)
        shapes = torch.Size((3, 32, 32)), 10
    elif name == "cifar100":
        dataset = datasets.CIFAR100
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        shapes = torch.Size((3, 32, 32)), 100
    else:
        raise ValueError(f"unknown dataset '{name}'")

    pre_processing = transforms.Compose([transforms.ToTensor()])
    train_data = dataset(root, train=True, transform=pre_processing)
    test_data = dataset(root, train=False, transform=pre_processing)
    if pre_process == "normal":
        pre_processing.transforms.append(transforms.Normalize(mean, std))
    elif pre_process == "pca":
        pre_processing.transforms.append(Whiten(train_data, zca=False))
    elif pre_process == "zca":
        pre_processing.transforms.append(Whiten(train_data, zca=True))

    return (train_data, test_data), shapes


def run(hparams, sys_config, log_dir):
    make_deterministic(hparams.seed)

    (train, test), shapes = get_data(root=sys_config.data_root, **hparams.data)
    train_loader = DataLoader(train, shuffle=True, batch_size=hparams.batch_size,
                              num_workers=sys_config.num_workers)
    test_loader = DataLoader(test, batch_size=len(test),
                              num_workers=sys_config.num_workers)

    device = sys_config.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(*shapes, **hparams.model).to(device)

    trainer = Trainer(
        model=model,
        objective=nn.CrossEntropyLoss(reduction="sum"),
        optimiser=torch.optim.Adam(model.parameters(), **hparams.adam),
        tb_writer=SummaryWriter(log_dir),
    )

    # logging
    save_config(hparams, f"{log_dir}/config.yaml")

    results = trainer.train(train_loader, test_loader, hparams.num_epochs)
    trainer.logger.close()
    return results["acc"]


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str, default="~/.pytorch")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--parallel", "-P", type=int, default=5)
    hparams, sys_config = upsilonconf.from_cli(parser=parser)
    repetitions = 10

    run_id = hparams.pop("id")
    stamp = time.strftime("%y%j-%H%M%S")

    def _run(hpars):
        log_path = Path("results", "mlp_search", "_".join([hpars.id, stamp]))
        try:
            return run(hpars, sys_config, log_path), log_path
        except RuntimeError as err:
            print(f"run {hpars.id} failed:", err)
            return float("nan"), None

    rng = random.Random(hparams.pop("seed"))
    seeds = [rng.randint(10_000, 100_000) for _ in range(repetitions)]
    pool = ProcessPoolExecutor(max_workers=sys_config.parallel)
    accuracies = pool.map(_run, [
        hparams | {"seed": s, "id": ".".join([run_id, str(i)])}
        for i, s in enumerate(seeds)])
    print([acc for acc, _ in accuracies])
    pool.shutdown()
