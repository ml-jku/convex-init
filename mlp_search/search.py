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
from tensorboard.backend.event_processing import event_accumulator

from convex_modules import ConvexLinear, LazyClippedPositivity, LinearSkip, ExponentialPositivity, NoPositivity
from convex_init import TraditionalInitialiser, ConvexInitialiser
from pre_processing import Whiten
from trainer import Trainer
from utils import make_deterministic, lecun_init_, he_init_


def get_data(name: str, root: str, pre_process: str = "normal", train_split: float = 0.9):
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
    data = dataset(root, train=True, transform=pre_processing)
    if pre_process == "normal":
        pre_processing.transforms.append(transforms.Normalize(mean, std))
    elif pre_process == "pca":
        pre_processing.transforms.append(Whiten(data, zca=False))
    elif pre_process == "zca":
        pre_processing.transforms.append(Whiten(data, zca=True))

    num_train = int(train_split * len(data))
    return random_split(data, [num_train, len(data) - num_train]), shapes


def get_model(img_shape: torch.Size, num_classes: int, hidden: tuple = (),
              convex: str = None, fix_init: bool = True, skip: bool = False):
    """
    Create neural network for experiment.

    Parameters
    ----------
    img_shape : torch.Size
        The array shape of a single input image.
    num_classes : int
        The number of classes for the prediction task.
    hidden : tuple of ints, optional
        The number of neurons for each hidden layer in the network.
    convex : str, optional
        What type of convexity to use (none by default).
    fix_init : bool, optional
        Fix the initialisation of convex networks (true by default).
    skip : bool, optional
        Add skip-connections from the input to every layer (false by default).

    Returns
    -------
    model : torch.nn.Module
        Simple multi-layer perceptron, ready to train.
    """
    width = img_shape.numel()
    widths = (width, *hidden, num_classes)

    if convex is None or convex == "":
        positivity = NoPositivity()
    elif convex == "icnn":
        positivity = LazyClippedPositivity()
    elif convex == "exp":
        positivity = ExponentialPositivity()
    else:
        raise ValueError()

    # first layer is special
    layer1 = nn.Linear(width, widths[0])
    phi = nn.ReLU()
    layers = [layer1, *(
        nn.Sequential(phi, ConvexLinear(n_in, n_out, positivity=positivity))
        for n_in, n_out in zip(widths[:-1], widths[1:])
    )]

    lecun_init = TraditionalInitialiser()
    lecun_init(layer1.weight, layer1.bias)
    init = ConvexInitialiser() if (convex and fix_init) else TraditionalInitialiser(gain=2.)
    for seq in layers[1:]:
        init(seq[-1].weight, seq[-1].bias)

    if skip:
        skipped = LinearSkip(width, widths[1], nn.Sequential(*layers[:2]))
        for layer, num_out in zip(layers[2:], widths[2:]):
            skipped = LinearSkip(width, num_out, nn.Sequential(skipped, layer))
        layers = [skipped]

    return nn.Sequential(nn.Flatten(), *layers)


def run(hparams, sys_config):
    result_path = Path("results", "mlp_search")
    for p in result_path.glob(f"{hparams.id}_*"):
        if p.is_dir():
            try:
                event_path = next(p.glob(f"events.out.tfevents.*"))
                ea = event_accumulator.EventAccumulator(str(event_path))
                events = ea.Reload().Scalars("valid/acc")
                if len(events) == hparams.num_epochs + 1:
                    print(f"found existing results for {hparams.id}")
                    max_event = max(events, key=lambda e: e.value)
                    return max_event.step, max_event.value
                else:
                    print(f"run for {hparams.id} broke after {len(events)} epochs, re-running")
                    raise KeyError
            except KeyError:
                for file in p.iterdir():
                    file.unlink()

                p.rmdir()

    log_dir = result_path / time.strftime(f"{hparams.id}_%y%j-%H%M%S")
    make_deterministic(hparams.seed)

    (train, valid), shapes = get_data(root=sys_config.data_root, **hparams.data)
    train_loader = DataLoader(train, shuffle=True, batch_size=hparams.batch_size,
                              num_workers=sys_config.num_workers)
    valid_loader = DataLoader(valid, batch_size=len(valid),
                              num_workers=sys_config.num_workers)

    if sys_config.device is None:
        sys_config.overwrite("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(*shapes, **hparams.model).to(sys_config.device)

    trainer = Trainer(
        model=model,
        objective=nn.CrossEntropyLoss(reduction="sum"),
        optimiser=torch.optim.Adam(model.parameters(), **hparams.adam),
        tb_writer=SummaryWriter(log_dir),
    )

    # logging
    trainer.logger.add_graph(model, next(iter(train_loader))[0].to(sys_config.device))
    save_config(hparams, f"{log_dir}/config.yaml")
    tup = hparams.pop("model.hidden")
    hparams["model.hidden"] = torch.tensor(tup) if tup else torch.tensor([-1])
    trainer.log_hparams(hparams, {
        "valid/acc": float("nan"),
        "valid/loss": float("nan"),
    })
    hparams.overwrite("model.hidden", tup)  # reset for later usage

    results = trainer.train(train_loader, valid_loader, hparams.num_epochs)
    trainer.logger.close()
    return None, results["acc"]


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("config_dir", metavar="PATH", type=Path)
    parser.add_argument("--parallel", "-P", type=int, default=5)
    sys_config, ns = upsilonconf.from_cli(parser=parser)
    if not ns.config_dir.is_dir():
        raise ValueError(f"not a directory: {ns.config_dir}")

    def _run(conf_file):
        hparams = upsilonconf.load(conf_file)
        try:
            return run(hparams, sys_config), hparams
        except RuntimeError as err:
            print(f"run {hparams.id} failed:", err)
            return -1, hparams

    pool = ProcessPoolExecutor(max_workers=ns.parallel)
    accuracies = pool.map(_run, Path.glob(ns.config_dir, "*.yaml"), chunksize=16)

    best_acc = 0.
    best_params = None
    for (epoch, acc), hparams in accuracies:
        if epoch is not None:
            hparams.overwrite("num_epochs", epoch)
        if acc > best_acc:
            best_acc = acc
            best_params = hparams
    pool.shutdown()

    if best_params is not None:
        save_config(best_params, f"results/best_{best_params.id}.yaml")
        print(f"best config: {best_params.id} ({100 * best_acc:05.2f}%)")
        print(best_params)
