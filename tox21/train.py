from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from upsilonconf import Configuration, save_config

from convex_init import ConvexInitialiser, TraditionalInitialiser
from convex_modules import ConvexLinear, ExponentialPositivity, LazyClippedPositivity, LinearSkip, NoPositivity
from trainer import signal_propagation, Trainer
from utils import make_deterministic


class Tox21Original(Dataset):
    """
    Tox21 dataset as it was provided for the DeepTox challenge.

    This data can be downloaded from:
    http://bioinf.jku.at/research/DeepTox/tox21.html
    """

    def __init__(self, root: str, split: str = "train", normalize: bool = True):
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
    """
    Binary cross-entropy loss that masks out predictions for NaN values in the target tensor.
    """

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


def get_model(name: str, num_hidden: int = 128, bad_init: bool = False, skip: bool = False):
    num_in, num_out = 512, 12
    if name == "logreg":
        return nn.Linear(num_in, num_out)

    if name == "fc":
        positivity = NoPositivity()
    elif name == "convex":
        positivity = ExponentialPositivity()
    elif name == "icnn":
        positivity = LazyClippedPositivity()
    else:
        raise ValueError(f"unknown model name: {name}")

    model = nn.Sequential(
        nn.Dropout(p=.7),
        nn.Linear(num_in, num_hidden),
        nn.ReLU(),
        nn.Dropout(p=.5),
        ConvexLinear(num_hidden, num_hidden, positivity=positivity),
        nn.ReLU(),
        nn.Dropout(p=.5),
        ConvexLinear(num_hidden, num_out, positivity=positivity),
    )

    init = TraditionalInitialiser(gain=2.) if bad_init else ConvexInitialiser()
    for idx in [4, 7]:
        init(model[idx].weight, model[idx].bias)

    if skip:
        new_model = LinearSkip(num_in, num_hidden, model[1:5])
        new_model = LinearSkip(num_in, num_out, nn.Sequential(new_model, *model[5:]))
        model = nn.Sequential(model[0], new_model)

    return model


def run(hparams: Configuration, sys_config: Configuration, log_dir: Path):
    """
    Train Tox21 models for hyperparameter selection.

    The model is trained on the training data and validation metrics are logged to tensorboard.
    The weights of the model are never stored in this function
    """
    log_dir = Path(log_dir)
    make_deterministic(hparams.seed)

    # data
    train_data = Tox21Original(sys_config.data_root, split="train")
    valid_data = Tox21Original(sys_config.data_root, split="valid")
    train_loader = DataLoader(train_data, shuffle=True, batch_size=hparams.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

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
        tb_writer=SummaryWriter(str(log_dir)),
    )

    # logging
    trainer.logger.add_graph(model, next(iter(train_loader))[0].to(device))
    save_config(hparams, log_dir / "config.yaml")
    trainer.log_hparams(hparams, {
        "valid/acc": float("nan"),
        "valid/loss": float("nan"),
    })

    trainer.train(train_loader, valid_loader, hparams.epochs)
    trainer.logger.close()


def final_run(hparams: Configuration, sys_config: Configuration, log_dir: Path):
    """
    Train final Tox21 prediction model.

    The model is trained on the concatenation of training and validation data.
    Validation metrics are not logged, but the weights of the model will be stored (in a file).
    """
    log_dir = Path(log_dir)
    make_deterministic(hparams.seed)

    # data
    data = ConcatDataset([
        Tox21Original(sys_config.data_root, split="train"),
        Tox21Original(sys_config.data_root, split="valid"),
    ])
    train_loader = DataLoader(data, shuffle=True, batch_size=hparams.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(**hparams.model).to(device)

    # optimisation
    trainer = Trainer(
        model=model,
        objective=MaskedBCEWithLogitsLoss(reduction="sum"),
        optimiser=torch.optim.Adam(model.parameters(), **hparams.adam),
        tb_writer=SummaryWriter(str(log_dir)),
        checkpoint=True,
    )

    # logging
    save_config(hparams, log_dir / "config.yaml")

    results = trainer.train(train_loader, train_loader, hparams.epochs)
    trainer.logger.close()
    return results


if __name__ == "__main__":
    import time
    import random
    from upsilonconf import load_config, config_from_cli

    hparams = config_from_cli()
    sys_config = load_config(Path("config", "system", "local.yaml"))
    repetitions = 10

    stamp = time.strftime("%y%j-%H%M%S")
    rng = random.Random(hparams.pop("seed"))
    for i in range(repetitions):
        results = final_run(
            hparams | {"seed": rng.randint(1_000, 10_000)},
            sys_config,
            Path("results", "tox21", ".".join([stamp, str(i)]))
        )
