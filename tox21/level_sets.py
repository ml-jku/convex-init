from pathlib import Path

import numpy as np
import scipy
import torch
from torch import nn
from upsilonconf import load_config

from tox21.train import get_model, Tox21Original


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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("stamp")
    args = parser.parse_args()
    sys_config = load_config(Path("config", "system", "local.yaml"))
    result_path = Path("results", "tox21", args.stamp + ".0")
    hparams = load_config(result_path / "config.yaml")

    # data = Tox21Original(sys_config.data_root, split="train")
    # (x0, y0), (x1, y1) = data[416], data[4593]  # chosen by Johannes (at random)
    data = Tox21Original(sys_config.data_root, split="test")
    model = get_model(**hparams.model)
    model.load_state_dict(torch.load(result_path / "checkpoint.pt")["model"])
    model.eval().requires_grad_(False)
    num_steps = 8
    torch.manual_seed(hparams.seed)
    indices = torch.randint(len(data), size=(12, 2))
    for target_assay, (src_idx, target_idx) in enumerate(indices):
        (x0, y0), (x1, y1) = data[src_idx], data[target_idx]
        print(torch.sigmoid(model(torch.from_numpy(x0))))
        print(y0)
        print(torch.sigmoid(model(torch.from_numpy(x1))))
        print(y1)
        # (sketch of) approach from Nesterov et al.
        x_min = find_minimum(model, x0, assay_idx=target_assay)
        level_set = interpolate_along_level_set(
            model, x0, x1, x_min, assay_idx=target_assay
        )
        levels = model(torch.cat([level_set, torch.from_numpy(x1).unsqueeze(0)]))
        print(levels[:, target_assay])

        np.savez(
            file=str(result_path / f"level_set_task{target_assay + 1}.npz"),
            cddds=np.r_[level_set.numpy(), x1[None, :]],
            preds=torch.sigmoid(levels)
        )

        # my suggestion: use orthogonal projections
        x_ref = find_minimum(model, x0, assay_idx=target_assay, max_iter=10)
        direction = (x1 - x0) / np.linalg.norm(x1 - x0)
        level_set = track_level_set(
            model, x0, direction, x_ref, assay_idx=target_assay
        )
        levels = model(torch.cat([level_set, torch.from_numpy(x1).unsqueeze(0)]))
        print(levels[:, target_assay])

        np.savez(
            file=str(result_path / f"level_set_task{target_assay + 1}_.npz"),
            cddds=np.r_[level_set.numpy(), x1[None, :]],
            preds=torch.sigmoid(levels)
        )
