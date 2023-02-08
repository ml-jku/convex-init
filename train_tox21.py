import time
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
from torch import nn
from torch.utils.data import Dataset

from convex_modules import ConvexLinear, ExponentialPositivity, LazyClippedPositivity
from trainer import signal_propagation, Trainer
from utils import make_deterministic


class Tox21Original(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root).expanduser()
        path = self.root / "tox21_original"

        if split == "train" or split == "original_train":
            data = np.load(str(path / "train_data.npz"))
        elif split == "valid" or split == "original_valid":
            data = np.load(str(path / "valid_data.npz"))
        elif split == "test" or split == "original_test":
            data = np.load(str(path / "test_data.npz"))
        else:
            raise NotImplementedError("only original splits for now")

        descriptors = np.load(str(path / "compound_features_cddd.npy"))
        assays = data["labels"]
        labels = np.where((assays == 0) | (assays == 1), assays, np.nan)
        self.compounds = descriptors[data["compound_idx"]]
        self.labels = labels.astype(np.float32)

    def __getitem__(self, index: int):
        cddd = self.compounds[index]
        labels = self.labels[index]
        return cddd, labels

    def __len__(self):
        return len(self.labels)

    @property
    def positive_rates(self):
        return torch.from_numpy(np.nanmean(self.labels, axis=0))


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, input, target):
        mask = torch.isnan(target)
        raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.masked_fill(input, mask, .5),
            torch.masked_fill(target, mask, .5),
            weight=self.weight, pos_weight=self.pos_weight, reduction="none"
        )

        masked_bce = torch.masked_fill(raw_bce, mask, torch.nan)
        if self.reduction == "sum":
            return torch.nansum(masked_bce)
        elif self.reduction == "mean":
            return torch.nanmean(masked_bce)
        else:
            return masked_bce


def get_model(name: str, num_hidden: int = 128):
    num_in, num_out = 512, 12
    if name == "logreg":
        return nn.Linear(num_in, num_out)
    elif name == "fc":
        return nn.Sequential(
            nn.Dropout(p=.7),
            nn.Linear(num_in, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(num_hidden, num_out),
        )
    elif name == "convex":
        model = nn.Sequential(
            nn.Dropout(p=.7),
            nn.Linear(num_in, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.5),
            ConvexLinear(num_hidden, num_hidden, positivity=ExponentialPositivity()),
            nn.ReLU(),
            nn.Dropout(p=.5),
            ConvexLinear(num_hidden, num_out, positivity=ExponentialPositivity()),
        )
        return model
    elif name == "icnn":
        model = nn.Sequential(
            nn.Dropout(p=.7),
            nn.Linear(num_in, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.5),
            ConvexLinear(num_hidden, num_hidden, positivity=LazyClippedPositivity()),
            nn.ReLU(),
            nn.Dropout(p=.5),
            ConvexLinear(num_hidden, num_out, positivity=LazyClippedPositivity()),
        )
        return model
    else:
        raise ValueError(f"unknown model name: {name}")


def find_minimum(model: nn.Module, start: torch.Tensor, assay_idx: int = 0,
                 max_iter: int = 1000, step_size: float = 1.):
    device = next(model.parameters()).device
    x_min = torch.as_tensor(start).clone().requires_grad_().to(device)
    for _ in range(max_iter):
        min_level = model(x_min)[assay_idx]
        with torch.no_grad():
            x_min -= step_size * torch.autograd.grad(min_level, x_min)[0]

    return x_min.detach()


def interpolate_along_level_set(model, x0, x1, x_min,
                                num_steps: int = 10, assay_idx: int = 0):
    device = next(model.parameters()).device
    x_start = torch.as_tensor(x0).to(device)
    x_end = torch.as_tensor(x1).to(device)
    target_level = model(x_start)[assay_idx]
    steps = torch.linspace(0., 1., num_steps + 2)[1:-1].unsqueeze(1).to(device)
    x_mid = (x_start + steps * (x_end - x_start))
    radii = torch.tensor([scipy.optimize.minimize_scalar(
        lambda r: abs(model(x_midi + r * (x_min - x_midi))[assay_idx] - target_level).cpu()
    ).x for x_midi in x_mid]).float().to(device)
    return torch.cat(
        [x_start.unsqueeze(0), x_mid + radii.unsqueeze(1) * (x_min - x_mid)]
    )


def track_level_set(model, x0, direction, x_ref,
                    num_steps: int = 10, assay_idx: int = 0):
    device = next(model.parameters()).device
    direction = torch.as_tensor(direction).to(device)
    x_curr = torch.as_tensor(x0).requires_grad_().to(device)
    target_level = model(x_curr)[assay_idx]
    g_curr, = torch.autograd.grad(target_level, x_curr)
    target_level = target_level.detach()

    points = [x_curr.detach()]
    for _ in range(num_steps):
        projected_dir = direction - g_curr * (g_curr @ direction) / (g_curr @ g_curr)
        seed_x = x_curr.detach() + projected_dir
        res = scipy.optimize.minimize_scalar(
            lambda alpha: abs(model(seed_x + alpha * (x_ref - seed_x))[assay_idx] - target_level).cpu(),
            bounds=(0., 1.)
        )
        x_curr = (seed_x + res.x * (x_ref - seed_x)).requires_grad_()
        g_curr, = torch.autograd.grad(model(x_curr)[assay_idx], x_curr)
        points.append(x_curr.detach())

    return torch.stack(points)


if __name__ == "__main__":
    from upsilonconf import load_config, save_config
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    config = load_config("config")
    hparams, sys_config = config.hparams, config.system
    log_dir = time.strftime("runs/tox21/%y%j-%H%M%S")
    make_deterministic(hparams.seed)

    # data
    train = Tox21Original(sys_config.data_root, split="train")
    valid = Tox21Original(sys_config.data_root, split="valid")
    train_loader = DataLoader(train, shuffle=True, batch_size=hparams.batch_size)
    valid_loader = DataLoader(valid, batch_size=len(valid))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(**hparams.model).to(device)

    means, varis = signal_propagation(model, torch.randn(1000, 512).cuda())
    print(model)
    print("squared means: ", means)
    print("variances:     ", varis)
    print("second moments:", [tuple(vi + mi for vi, mi in zip(v, m))
                              for v, m in zip(varis, means)])
    # raise RuntimeError("stopping here")

    # optimisation
    trainer = Trainer(
        model=model,
        objective=MaskedBCEWithLogitsLoss(reduction="sum"),
        optimiser=torch.optim.Adam(model.parameters(), **hparams.adam),
        tb_writer=SummaryWriter(log_dir),
    )

    # logging
    trainer.logger.add_graph(model, next(iter(train_loader))[0].to(device))
    save_config(hparams, log_dir + "/config.yaml")
    trainer.log_hparams(hparams, {
        "valid/acc": float("nan"),
        "valid/loss": float("nan"),
    })

    trainer.train(train_loader, valid_loader, hparams.epochs)
    trainer.logger.close()


    target_assay, num_steps = 0, 10
    (x0, _), (x1, _) = valid[0], valid[1]
    model.eval().requires_grad_(False)

    # (sketch of) approach from Nesterov et al.
    x_min = find_minimum(model, x0, assay_idx=target_assay)
    level_set = interpolate_along_level_set(
        model, x0, x1, x_min, assay_idx=target_assay
    )
    levels = model(level_set)[:, target_assay]
    print(levels)

    # my suggestion: use orthogonal projections
    x_ref = find_minimum(model, x0, assay_idx=target_assay, max_iter=10)
    direction = (x1 - x0) / np.linalg.norm(x1 - x0)
    level_set = track_level_set(
        model, x0, direction, x_ref, assay_idx=target_assay
    )
    levels = model(level_set)[:, target_assay]
    print(levels)
