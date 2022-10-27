import os
import time

import torch
from torch import nn

from convex_modules import *
from trainer import Trainer, signal_propagation


def get_model(name: str, hidden: tuple[int] = (), positivity: str = "exp", init: str = "he"):
    in_shape, num_classes = torch.Size((1, 28, 28)), 10
    layer_sizes = (*hidden, num_classes)

    layer1 = nn.Linear(in_shape.numel(), layer_sizes[0])
    nn.init.kaiming_normal_(layer1.weight, nonlinearity="linear")
    nn.init.zeros_(layer1.bias)

    pos_func = ExponentialPositivity() if positivity == "exp" else ClippedPositivity()

    name = name.strip().lower()
    if name == "regression":
        return nn.Sequential(nn.Flatten(), layer1)
    if name == "mlp":
        layers = [nn.Sequential(
            nn.ReLU(), nn.Linear(n_in, n_out)
        ) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        for lay in layers:
            nn.init.kaiming_normal_(lay[1].weight, nonlinearity="relu")
            nn.init.zeros_(lay[1].bias)
        return nn.Sequential(nn.Flatten(), layer1, *layers)
    if name == "convex-mlp":
        layers = [nn.Sequential(
            nn.ReLU(), ConvexLinear(n_in, n_out, positivity=pos_func)
        ) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        for lay in layers:
            if init == "he":
                nn.init.kaiming_normal_(lay[-1].weight)
                nn.init.zeros_(lay[-1].bias)
            else:
                pos_func.init_raw_(lay[-1].weight, lay[-1].bias)
        return nn.Sequential(nn.Flatten(), layer1, *layers)
    else:
        raise ValueError(f"unknown model name: '{name}'")


def make_deterministic(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


if __name__ == "__main__":
    from upsilonconf import load_config, save_config
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    from torch.utils.tensorboard import SummaryWriter

    config = load_config("config")
    hparams, sys_config = config.values()
    log_dir = time.strftime("runs/mnist/%y%j-%H%M%S")
    make_deterministic(hparams.seed)

    # data
    data = datasets.MNIST(sys_config.data_root, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))
    ]))
    train, valid = random_split(data, [55_000, 5_000])
    train_loader = DataLoader(train, shuffle=True, batch_size=hparams.batch_size,
                              num_workers=sys_config.num_workers)
    valid_loader = DataLoader(valid, batch_size=len(valid),
                              num_workers=sys_config.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(**hparams.model).to(device)

    means, varis = signal_propagation(model, torch.randn(1000, 1, 28, 28).cuda())
    print(model)
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

    # setup logging
    trainer.logger.add_graph(model, next(iter(train_loader))[0].to(device))
    save_config(hparams, log_dir + "/config.yaml")
    hparams.model.overwrite("hidden", str(hparams.model.hidden))
    trainer.log_hparams(hparams, {f"valid/acc": float("nan")})

    trainer.train(train_loader, valid_loader, hparams.epochs)
    trainer.logger.close()
