import os
import time

import torch
from torch import nn

from convex_modules import *
from convex_modules import BiConvex
from trainer import Trainer, signal_propagation, make_deterministic


def get_model(name: str, hidden: tuple[int] = (), positivity: str = "exp", init: str = "he"):
    in_shape, num_classes = torch.Size((3, 32, 32)), 10
    layer_sizes = (*hidden, num_classes)

    layer1 = nn.Linear(in_shape.numel(), layer_sizes[0])
    nn.init.kaiming_normal_(layer1.weight, nonlinearity="linear")
    nn.init.zeros_(layer1.bias)

    pos_func = ExponentialPositivity() if positivity == "exp" else ClippedPositivity()

    name = name.strip().lower()
    if name == "regression":
        return nn.Sequential(nn.Flatten(), layer1)
    elif name == "mlp":
        layers = [nn.Sequential(
            nn.ReLU(), nn.Linear(n_in, n_out)
        ) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        for lay in layers:
            nn.init.kaiming_normal_(lay[1].weight, nonlinearity="relu")
            nn.init.zeros_(lay[1].bias)
        return nn.Sequential(nn.Flatten(), layer1, *layers)
    elif name == "convex-mlp":
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
    elif name == "biconvex-mlp":
        layers = [nn.Sequential(
            nn.ReLU(), ConvexLinear(n_in, n_out, positivity=pos_func)
        ) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        for lay in layers:
            if init == "he":
                nn.init.kaiming_normal_(lay[-1].weight)
                nn.init.zeros_(lay[-1].bias)
            else:
                pos_func.init_raw_(lay[-1].weight, lay[-1].bias)
        return BiConvex(
            nn.Sequential(nn.Flatten(), layer1, *layers)
        )
    elif name == "cnn":
        if len(layer_sizes) != 4:
            raise ValueError(f"CNN must have three layers, but got {len(layer_sizes) - 1}")
        # layer1 = nn.Conv2d(in_shape[0], layer_sizes[0], 5)
        layer1 = nn.Conv2d(in_shape[0], layer_sizes[0], 5, stride=2)
        nn.init.kaiming_normal_(layer1.weight, nonlinearity="linear")
        nn.init.zeros_(layer1.bias)
        conv_layers = [nn.Sequential(
            # nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(n_in, n_out, 5)
            nn.ReLU(), nn.Conv2d(n_in, n_out, 5, stride=2)
        ) for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        for lay in conv_layers:
            nn.init.kaiming_normal_(lay[-1].weight)
            nn.init.zeros_(lay[-1].bias)
        fc_layers = [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(layer_sizes[-2], 300),
            nn.ReLU(),
            nn.Linear(300, layer_sizes[-1])
        ]
        for lay in fc_layers[2::2]:
            nn.init.kaiming_normal_(lay.weight)
            nn.init.zeros_(lay.bias)
        return nn.Sequential(layer1, *conv_layers, *fc_layers)
    elif name == "convex-cnn":
        if len(layer_sizes) != 4:
            raise ValueError(f"CNN must have three layers, but got {len(layer_sizes) - 1}")
        # layer1 = nn.Conv2d(in_shape[0], layer_sizes[0], 5)
        layer1 = nn.Conv2d(in_shape[0], layer_sizes[0], 5, stride=2)
        nn.init.kaiming_normal_(layer1.weight, nonlinearity="linear")
        nn.init.zeros_(layer1.bias)
        conv_layers = [nn.Sequential(
            # nn.MaxPool2d(2), nn.ReLU(), ConvexConv2d(n_in, n_out, 5, positivity=pos_func)
            nn.ReLU(), ConvexConv2d(n_in, n_out, 5, stride=2, positivity=pos_func)
        ) for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        for lay in conv_layers:
            if init == "he":
                nn.init.kaiming_normal_(lay[-1].weight)
                nn.init.zeros_(lay[-1].bias)
            elif init == "hacked":
                assert lay[-1].weight.shape == (128, 128, 5, 5)
                nn.init.normal_(lay[-1].weight, -10.872, 1.05 * 2.615)
                nn.init.constant_(lay[-1].bias, -0.740)
            else:
                pos_func.init_raw_(lay[-1].weight, lay[-1].bias)
        fc_layers = [
            nn.Flatten(),
            nn.ReLU(),
            ConvexLinear(layer_sizes[-2], 300, positivity=pos_func),
            nn.ReLU(),
            ConvexLinear(300, layer_sizes[-1], positivity=pos_func)
        ]
        for lay in fc_layers[2::2]:
            if init == "he":
                nn.init.kaiming_normal_(lay.weight)
                nn.init.zeros_(lay.bias)
            else:
                pos_func.init_raw_(lay.weight, lay.bias)
        return nn.Sequential(layer1, *conv_layers, *fc_layers)
    elif name == "biconvex-cnn":
        if len(layer_sizes) != 4:
            raise ValueError(f"CNN must have three layers, but got {len(layer_sizes) - 1}")
        layer1 = nn.Conv2d(in_shape[0], layer_sizes[0], 5)
        nn.init.kaiming_normal_(layer1.weight, nonlinearity="linear")
        nn.init.zeros_(layer1.bias)
        conv_layers = [nn.Sequential(
            nn.MaxPool2d(2), nn.ReLU(), ConvexConv2d(n_in, n_out, 5, positivity=pos_func)
        ) for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        for lay in conv_layers:
            if init == "he":
                nn.init.kaiming_normal_(lay[-1].weight)
                nn.init.zeros_(lay[-1].bias)
            else:
                pos_func.init_raw_(lay[-1].weight, lay[-1].bias)
        fc_layers = [
            nn.Flatten(),
            nn.ReLU(),
            ConvexLinear(layer_sizes[-2], 300, positivity=pos_func),
            nn.ReLU(),
            ConvexLinear(300, layer_sizes[-1], positivity=pos_func)
        ]
        for lay in fc_layers[2::2]:
            if init == "he":
                nn.init.kaiming_normal_(lay.weight)
                nn.init.zeros_(lay.bias)
            else:
                pos_func.init_raw_(lay.weight, lay.bias)
        model = BiConvex(
            nn.Sequential(layer1, *conv_layers, *fc_layers)
        )
        nn.init.kaiming_normal_(model.conv_net[0].weight, nonlinearity="linear")
        nn.init.zeros_(model.conv_net[0].bias)
        for lay in model.conv_net[1:len(conv_layers) + 1]:
            pos_func.init_raw_(lay[-1].weight, lay[-1].bias)
        for lay in model.conv_net[len(conv_layers) + 3::2]:
            pos_func.init_raw_(lay.weight, lay.bias)
        return model
    else:
        raise ValueError(f"unknown model name: '{name}'")


if __name__ == "__main__":
    from upsilonconf import load_config, save_config
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    from torch.utils.tensorboard import SummaryWriter

    config = load_config("config")
    hparams, sys_config = config.values()
    log_dir = time.strftime("runs/cifar/%y%j-%H%M%S")
    make_deterministic(hparams.seed)

    # data
    data = datasets.CIFAR10(sys_config.data_root, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, .2435, 0.2616))
    ]))
    train, valid = random_split(data, [45_000, 5_000])
    train_loader = DataLoader(train, shuffle=True, batch_size=hparams.batch_size,
                              num_workers=sys_config.num_workers)
    valid_loader = DataLoader(valid, batch_size=len(valid),
                              num_workers=sys_config.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(**hparams.model).to(device)

    means, varis = signal_propagation(model, torch.randn(1000, 3, 32, 32).cuda())
    print(model)
    print("squared means: ", means)
    print("variances:     ", varis)
    print("second moments:", [tuple(vi + mi for vi, mi in zip(v, m))
                              for v, m in zip(varis, means)])
    # raise RuntimeError("stopping here")

    # optimisation
    trainer = Trainer(
        model=model,
        objective=nn.CrossEntropyLoss(reduction="sum"),
        optimiser=torch.optim.Adam(model.parameters(), **hparams.adam),
        tb_writer=SummaryWriter(log_dir),
    )

    # logging
    trainer.logger.add_graph(model, next(iter(train_loader))[0].to(device))
    save_config(hparams, log_dir + "/config.yaml")
    hparams.model.overwrite("hidden", str(hparams.model.hidden))
    trainer.log_hparams(hparams, {
        "valid/acc": float("nan"),
        "valid/loss": float("nan"),
    })

    trainer.train(train_loader, valid_loader, hparams.epochs)
    trainer.logger.close()
