"""
Experiment to test learnability of input-convex net as function of depth.
"""
import time
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from upsilonconf import Configuration, load_config, save_config

from convex_modules import *
from trainer import make_deterministic, signal_propagation, Trainer


def lecun_init_(w: torch.Tensor, b: torch.Tensor) -> None:
    """
    Initialise weights according to (Lecun et al., 1998).

    Parameters
    ----------
    w : torch.Tensor
        Weight matrix to initialise.
    b : torch.Tensor
        Bias vector to initialise.
    """
    nn.init.kaiming_normal_(w, nonlinearity="linear")
    nn.init.zeros_(b)


def he_init_(w: torch.Tensor, b: torch.Tensor) -> None:
    """
    Initialise weights according to (He et al., 2015).

    Parameters
    ----------
    w : torch.Tensor
        Weight matrix to initialise.
    b : torch.Tensor
        Bias vector to initialise.
    """
    nn.init.kaiming_normal_(w, nonlinearity="relu")
    nn.init.zeros_(b)


def get_layer(n_in: int, n_out: int,
              positivity: Positivity = None,
              better_init: bool = True):
    """
    Create fully-connected layer.

    Parameters
    ----------
    n_in : int
        The number of input features.
    n_out : int
        The number of output features.
    positivity : Positivity, optional
        The positivity object to use for the convex layer.
        If ``None``, the layer will be non-convex.
    better_init : bool, optional
        Use better initialisation than the default (He et al., 2015) if possible.

    Returns
    -------
    layer : torch.nn.Module
        A fully-connected network layer (without activation function).
    """
    if positivity is None:
        layer = nn.Linear(n_in, n_out)
        he_init_(layer.weight, layer.bias)
        return layer

    _init = positivity.init_raw_ if better_init else he_init_
    layer = ConvexLinear(n_in, n_out, positivity=positivity)
    _init(layer.weight, layer.bias)
    return layer


def get_model(img_shape: torch.Size, num_classes: int,
              num_hidden: int = 1,
              positivity: str = None,
              better_init: bool = True):
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
         - ``""`` or ``None`` results in a NON-convex network
    better_init : bool, optional
        Use better initialisation than the default (He et al., 2015) if possible.

    Returns
    -------
    model : torch.nn.Module
        Simple multi-layer perceptron, ready to train.
    """
    width = img_shape.numel()
    widths = (width, ) * num_hidden + (num_classes, )

    # first layer is special
    layer1 = nn.Linear(width, widths[0])
    lecun_init_(layer1.weight, layer1.bias)

    if positivity == "":
        positivity = None
    elif positivity == "exp":
        positivity = ExponentialPositivity()
    elif positivity == "clip":
        positivity = ClippedPositivity()
    elif positivity is not None:
        raise ValueError(f"unknown value for positivity: '{positivity}'")

    phi = nn.ReLU()
    return nn.Sequential(nn.Flatten(), layer1, *(
        nn.Sequential(phi, get_layer(n_in, n_out, positivity, better_init))
        for n_in, n_out in zip(widths[:-1], widths[1:])
    ))


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


def run(hparams: Configuration, sys_config: Configuration):
    """ Run experiment with given parameters. """
    log_dir = time.strftime("results/init_learnability/%y%j-%H%M%S")
    make_deterministic(hparams.seed)

    # data
    (train, valid), shapes = get_data(**hparams.data, root=sys_config.data_root)
    train_loader = DataLoader(train, shuffle=True, batch_size=hparams.batch_size,
                              num_workers=sys_config.num_workers)
    valid_loader = DataLoader(valid, batch_size=len(valid),
                              num_workers=sys_config.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(*shapes, **hparams.model).to(device)

    means, varis = signal_propagation(model, torch.randn(1000, *shapes[0]).cuda())
    print("squared means: ", means)
    print("variances:     ", varis)
    print("second moments:", [tuple(vi + mi for vi, mi in zip(v, m))
                              for v, m in zip(varis, means)])

    # optimisation
    trainer = Trainer(
        model=model,
        objective=nn.CrossEntropyLoss(reduction="sum"),
        optimiser=torch.optim.Adam(model.parameters(), **hparams.adam),
        tb_writer=SummaryWriter(log_dir),
    )

    # logging
    trainer.logger.add_graph(model, next(iter(train_loader))[0].to(device))
    save_config(hparams, f"{log_dir}/config.yaml")
    trainer.log_hparams(hparams, {
        "valid/acc": float("nan"),
        "valid/loss": float("nan"),
    })

    results = trainer.train(train_loader, valid_loader, hparams.epochs)
    torch.save(model.state_dict(), f"{log_dir}/checkpoint.pth")
    trainer.logger.close()
    return results


if __name__ == "__main__":
    from random import Random
    from upsilonconf import config_from_cli

    hparams = config_from_cli()
    system_config = load_config("config/system.yaml")

    rng = Random(hparams.seed)
    for seed in (rng.randint(1_000, 10_000) for _ in range(10)):
        hparams.overwrite("seed", seed)
        print(hparams)
        run(hparams, system_config)
