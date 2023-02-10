from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from upsilonconf import Configuration, load_config

from tox21.train import get_model, Tox21Original
from trainer import AreaUnderROCCurve


def run(log_dir: Path, sys_config: Configuration):
    log_dir = Path(log_dir)
    data = Tox21Original(sys_config.data_root, split="test")

    hparams = load_config(log_dir / "config.yaml")
    loader = DataLoader(data, batch_size=len(data))
    model = get_model(**hparams.model).eval()
    model.load_state_dict(torch.load(log_dir / "checkpoint.pt")["model"])
    metrics = {f"auc{i}": AreaUnderROCCurve(task=i) for i in range(12)}
    device = next(model.parameters()).device

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    for k, auc in metrics.items():
        auc(logits, y)

    return {k: auc.value for k, auc in metrics.items()}


if __name__ == "__main__":
    stamp = "23041-203625"
    sys_config = load_config(Path("config", "system", "local.yaml"))
    result_path = Path("results", "tox21", stamp)

    results = {}
    for rep in result_path.parent.glob(".".join([stamp, "*"])):
        rep_results = run(rep, sys_config)
        for k, v in rep_results.items():
            results.setdefault(k, []).append(v)

    averages = []
    for k, v in results.items():
        low, mid, high = 100 * np.quantile(v, [0.05, 0.5, 0.95])
        averages.append(np.mean(v))
        print(f"{k:10s} {mid:5.2f}({low:5.2f} - {high:5.2f}) +- {max(high - mid, mid - low):5.2f}")
    low, mid, high = 100 * np.quantile(averages, [0.05, 0.5, 0.95])
    print(f"{'avg_auc':10s} {mid:5.2f}({low:5.2f} - {high:5.2f}) +- {max(high - mid, mid - low):5.2f}")
