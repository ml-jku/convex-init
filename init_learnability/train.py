"""
Experiment to test learnability of input-convex net as function of depth.
"""
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from upsilonconf import Configuration, load_config, save_config

from convex_init import TraditionalInitialiser, ConvexInitialiser, ConvexBiasCorrectionInitialiser
from convex_modules import *
from trainer import Trainer
from utils import make_deterministic


def get_model(img_shape: torch.Size, num_classes: int,
              num_hidden: int = 1,
              positivity: str = None,
              better_init: bool = True,
              rand_bias: bool = False,
              corr: float = .5,
              skip: bool = False,
              bias_init_only: bool = False):
    """
    Create neural network for experiment.

    Parameters
    ----------
    img_shape : torch.Size
        The array shape of a single input image.
    num_classes : int
        The number of classes for the prediction task.
    num_hidden : int, optional
        The number of hidden layers in the network.
    positivity : str, optional
        The function to use to make weighs positive (for input-convex nets):
         - ``"exp"`` uses the exponential function to obtain positive weights
         - ``"clip"`` clips values at zero to obtain positive weights
         - ``"icnn"`` clips values at zero after each update
         - ``""`` or ``None`` results in a NON-convex network
    better_init : bool, optional
        Use principled initialisation for convex layers instead of default (He et al., 2015).
    rand_bias : bool, optional
        Use random bias initialisation instead of constants for convex nets.
    corr : float, optional
        The correlation fixed point to aim for in the better initialisation.
    skip : bool, optional
        Wrap layer in skip-connection.
    bias_init_only: bool, optional
        Only apply principled initialisation for bias parameters.
        Weight parameters are initialised using the default (He et al., 2015) initialisation.

    Returns
    -------
    model : torch.nn.Module
        Simple multi-layer perceptron, ready to train.
    """
    width = img_shape.numel()
    widths = (width, ) * num_hidden + (num_classes, )

    if positivity is None or positivity == "":
        positivity = NoPositivity()
    elif positivity == "exp":
        positivity = ExponentialPositivity()
    elif positivity == "clip":
        positivity = ClippedPositivity()
    elif positivity == "icnn":
        positivity = LazyClippedPositivity()
    elif positivity is not None:
        raise ValueError(f"unknown value for positivity: '{positivity}'")

    # first layer can be regular
    layer1 = nn.Linear(width, widths[0])
    phi = nn.ReLU()
    layers = [layer1, *(
        nn.Sequential(phi, ConvexLinear(n_in, n_out, positivity=positivity))
        for n_in, n_out in zip(widths[:-1], widths[1:])
    )]

    # initialisation
    lecun_init = TraditionalInitialiser(gain=1.)
    if better_init and not isinstance(positivity, NoPositivity):
        if bias_init_only:
            init = ConvexBiasCorrectionInitialiser(positivity, gain=2.)
        else:
            init = ConvexInitialiser(var=1., corr=corr, bias_noise=0.5 if rand_bias else 0.)
    else:
        init = TraditionalInitialiser(gain=2.)

    lecun_init(layer1.weight, layer1.bias)
    for _, convex_layer in layers[1:]:
        init(convex_layer.weight, convex_layer.bias)

    if skip:
        skipped = LinearSkip(width, widths[1], nn.Sequential(*layers[:2]))
        for layer, num_out in zip(layers[2:], widths[2:]):
            skipped = LinearSkip(width, num_out, nn.Sequential(skipped, layer))
        layers = [skipped]

    return nn.Sequential(nn.Flatten(), *layers)


def get_data(name: str, root: str, train_split: float = 0.9):
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

    data = dataset(root, train=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ]))

    num_train = int(train_split * len(data))
    return random_split(data, [num_train, len(data) - num_train]), shapes


def run(hparams: Configuration, sys_config: Configuration, log_dir: Path | str):
    """ Run experiment with given parameters. """
    log_dir = Path(log_dir)
    make_deterministic(hparams.seed)

    # data
    (train, valid), shapes = get_data(**hparams.data, root=sys_config.data_root)
    train_loader = DataLoader(train, shuffle=True, batch_size=hparams.batch_size,
                              num_workers=sys_config.num_workers)
    valid_loader = DataLoader(valid, batch_size=len(valid),
                              num_workers=sys_config.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(*shapes, **hparams.model).to(device)

    # means, varis = signal_propagation(model, torch.randn(1000, *shapes[0]).cuda())
    # print("squared means: ", means)
    # print("variances:     ", varis)
    # print("second moments:", [tuple(vi + mi for vi, mi in zip(v, m))
    #                           for v, m in zip(varis, means)])

    # optimisation
    trainer = Trainer(
        model=model,
        objective=nn.CrossEntropyLoss(reduction="sum"),
        optimiser=torch.optim.Adam(model.parameters(), **hparams.adam),
        tb_writer=SummaryWriter(str(log_dir)),
    )

    # logging
    trainer.logger.add_graph(model, next(iter(train_loader))[0].to(device))
    save_config(hparams, log_dir / "config.yaml")
    trainer.log_hparams(hparams, {
        "valid/acc": float("nan"),
        "valid/loss": float("nan"),
    })

    results = trainer.train(train_loader, valid_loader, hparams.num_epochs)
    trainer.logger.close()
    return results


if __name__ == "__main__":
    from random import Random
    from upsilonconf import config_from_cli

    hparams = config_from_cli()
    system_config = load_config("config/system/local.yaml")
    repetitions = 10

    stamp = time.strftime("%y%j-%H%M%S")
    rng = Random(hparams.pop("seed"))
    for i in range(repetitions):
        run(
            hparams | {"seed": rng.randint(1_000, 10_000)},
            system_config,
            Path("results", "init_learnability", f"{stamp}.{i:d}")
        )
