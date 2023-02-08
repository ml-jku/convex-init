from pathlib import Path

import torch
from sklearn import metrics
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm import tqdm
from upsilonconf import Configuration


class Accuracy:

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(correct={self.correct}, count={self.count})"

    def __str__(self):
        return f"{100 * self.value:05.2f}% (on {self.count} samples)"

    def __call__(self, logits, y):
        num_correct = logits.argmax(dim=1).eq(y).sum(0).item()
        batch_size = len(y)
        self.correct += num_correct
        self.count += batch_size
        return num_correct / batch_size

    @property
    def value(self) -> float:
        return self.correct / self.count if self.count > 0 else float("nan")

    def reset(self):
        self.correct = 0
        self.count = 0
        return self


class AreaUnderROCCurve:

    def __init__(self, task: int = None):
        self.task = task
        self.pairs = []
        self._cache = None

    def __str__(self):
        return f"{100 * self.value:05.2f}% (on {len(self.pairs)} samples)"

    def __call__(self, logits, y):
        if self.task is not None:
            logits, y = logits[..., self.task], y[..., self.task]

        mask = (y == 0) | (y == 1)
        logits = torch.masked_select(logits, mask)
        y = torch.masked_select(y, mask)
        self.pairs.extend(zip(logits.tolist(), y.tolist()))
        self._cache = None

    @property
    def value(self) -> float:
        y_score, y_true = zip(*self.pairs)
        return metrics.roc_auc_score(y_true, y_score)

    def reset(self):
        self.pairs = []
        self._cache = None
        return self


class Average:

    def __init__(self):
        self.total = 0.
        self.count = 0

    def __str__(self):
        return f"{self.value:.3f} (avg over {self.count} samples)"

    def __call__(self, val: float, batch_size: int) -> float:
        self.total += val
        self.count += batch_size
        return val / batch_size

    @property
    def value(self) -> float:
        return self.total / self.count if self.count > 0 else float("nan")

    def reset(self):
        self.total = 0.
        self.count = 0
        return self


class ExponentialMovingAverage:

    def __init__(self, func: nn.Module, alpha: float = 0.5, name: str = None):
        self._value = None
        self.alpha = alpha
        self.func = func
        self.name = func.__class__.__name__ if name is None else name

    def __str__(self):
        return f"{self.name} {self.alpha:.1f}-EMA: {self.value:.3f}"

    def __call__(self, logits, y):
        val = self.func(logits, y)
        if self._value is None:
            self._value = val.item() / len(y)
            return val

        self._value = (1 - self.alpha) * self._value + self.alpha * val.item() / len(y)
        return val

    @property
    def value(self) -> float:
        return float("nan") if self._value is None else self._value


def _iterate_layers(model):
    from convex_modules import BiConvex, LinearSkip
    if isinstance(model, nn.Sequential):
        for layer in model:
            yield from _iterate_layers(layer)
    elif isinstance(model, BiConvex):
        yield lambda x: (x, x)
        for layer1, layer2 in zip(
            _iterate_layers(model.conv_net),
            _iterate_layers(model.conc_net)
        ):
            yield lambda x1, x2: layer1(x1) + layer2(x2)
        yield lambda x1, x2: (x1 - x2, )
    elif isinstance(model, LinearSkip):
        yield lambda x: (x, x)
        res_iter = _iterate_layers(model.residual)
        yield lambda x1, x2: (model.skip(x1), next(res_iter)(x2))
        for layer in res_iter:
            yield lambda x1, x2: (x1, layer(x2))
        yield lambda x1, x2: (x1 + x2, )
    else:
        yield lambda x: (model(x), )


def signal_propagation(model, inputs):
    x = (inputs, )
    mean = torch.mean(inputs.mean(0) ** 2).item()
    vari = torch.mean(inputs.var(0)).item()
    means, varis = [(mean, )], [(vari, )]
    _flat = inputs.view(len(inputs), -1)
    cov_diag = torch.mean(_flat ** 2, dim=0).mean()
    cov_rest = sum(
        torch.mean(_flat * _flat.roll(i, dims=-1), dim=0)[i:].sum()
        for i in range(1, _flat.shape[-1])
    ) * 2 / (_flat.shape[-1] * (_flat.shape[-1] - 1))
    print(cov_diag.item(), cov_rest.item())

    for layer in _iterate_layers(model):
        x = layer(*x)

        for xi in x:
            _flat = xi.view(len(xi), -1)
            cov_diag = torch.mean(_flat ** 2, dim=0).mean()
            cov_diag1 = torch.sum(_flat[1:] * _flat.roll(1, dims=-1)[1:], dim=0).mean() / len(xi)
            cov_diag2 = torch.sum(_flat[2:] * _flat.roll(2, dims=-1)[2:], dim=0).mean() / len(xi)
            cov_diag3 = torch.sum(_flat[3:] * _flat.roll(3, dims=-1)[3:], dim=0).mean() / len(xi)
            cov_diag4 = torch.sum(_flat[4:] * _flat.roll(4, dims=-1)[4:], dim=0).mean() / len(xi)
            cov_diag5 = torch.sum(_flat[5:] * _flat.roll(5, dims=-1)[5:], dim=0).mean() / len(xi)
            cov_diag6 = torch.sum(_flat[6:] * _flat.roll(6, dims=-1)[6:], dim=0).mean() / len(xi)
            print(cov_diag.item(), cov_diag1.item(), cov_diag2.item(),
                  cov_diag3.item(), cov_diag4.item(), cov_diag5.item(), cov_diag6.item())

        means.append(
            tuple(torch.mean(xi.mean(0) ** 2).item() for xi in x)
        )
        varis.append(
            tuple(torch.mean(xi.var(0)).item() for xi in x)
        )

    return means, varis


class Trainer:

    def __init__(
        self, model: nn.Module, objective: nn.Module, optimiser: optim.Optimizer,
            tb_writer: SummaryWriter = None
    ):
        device = next(model.parameters()).device
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.optimiser = optimiser

        self.logger = tb_writer
        self.num_epochs = 0
        self.num_updates = 0

    def log_hparams(self, config: Configuration, metrics: dict):
        exp, ssi, sei = hparams(config.to_dict(flat=True), metrics, None)
        self.logger.file_writer.add_summary(exp)
        self.logger.file_writer.add_summary(ssi)
        self.logger.file_writer.add_summary(sei)

    def _forward(self, batch):
        device = next(self.model.parameters()).device
        x, y = (t.to(device) for t in batch)
        return self.model(x), y

    @torch.no_grad()
    def evaluate(self, batches, metrics=None):
        self.model.eval()
        self.objective.eval()

        if metrics is None:
            metrics = {}

        metrics = {k: m.reset() for k, m in metrics.items()}
        avg_loss = Average()
        for batch in tqdm(batches, total=len(batches), desc="evaluating"):
            logits, y = self._forward(batch)
            err = self.objective(logits, y)
            avg_loss(err, len(y))
            [m(logits, y) for m in metrics.values()]

        metrics["loss"] = avg_loss
        return metrics

    def update(self, batches):
        self.model.train()
        self.objective.train()

        avg_loss = Average()
        for batch in tqdm(batches, total=len(batches), desc="updating"):
            logits, y = self._forward(batch)
            err = self.objective(logits, y)
            if self.logger is not None:
                self.logger.add_scalar("train/batch_loss", avg_loss(err, len(y)), self.num_updates)

            self.optimiser.zero_grad()
            err.backward()
            self.optimiser.step()
            self.num_updates += 1

        self.num_epochs += 1
        return avg_loss.value

    def train(self, train_loader, valid_loader, num_epochs: int = 1):
        extra_metrics = {}
        if isinstance(self.objective, nn.CrossEntropyLoss):
            extra_metrics["acc"] = Accuracy()
        elif isinstance(self.objective, nn.BCEWithLogitsLoss):
            _, y = next(iter(valid_loader))
            assert y.ndim > 1
            extra_metrics.update({
                f"auc{i}": AreaUnderROCCurve(i) for i in range(y.shape[-1])
            })

        # baseline
        out = self.evaluate(train_loader)
        metrics = self.evaluate(valid_loader, extra_metrics)
        if self.logger is not None:
            self.logger.add_scalar("train/avg_loss", out["loss"].value, self.num_epochs)
            for k, m in metrics.items():
                self.logger.add_scalar(f"valid/{k}", m.value, self.num_epochs)
        print(f"epoch {0:02d}",
              ", ".join(f"{k}: {v}" for k, v in metrics.items()),
              f"(avg train loss: {out['loss'].value:.5e})")
        auc_values = [v.value for k, v in metrics.items() if k.startswith('auc')]
        if len(auc_values) > 1:
            avg_auc = 100 * sum(auc_values) / len(auc_values)
            self.logger.add_scalar(f"valid/avg_auc", avg_auc, self.num_epochs)
            print(f"avg AUC: {avg_auc:05.2f}%")

        best_metrics = {k: m.value for k, m in metrics.items()}
        for epoch in range(1, num_epochs + 1):
            train_loss = self.update(train_loader)
            metrics = self.evaluate(valid_loader, extra_metrics)
            best_metrics = {
                k: min(metrics[k].value, v) if k != "acc"
                else max(metrics[k].value, v)
                for k, v in best_metrics.items()
            }

            if self.logger is not None:
                torch.save({
                    "model": self.model.state_dict(),
                    "optim": self.optimiser.state_dict(),
                    "epoch": self.num_epochs,
                }, Path(self.logger.log_dir) / "checkpoint.pt")

                self.logger.add_scalar("train/avg_loss", train_loss, self.num_epochs)
                for k, m in metrics.items():
                    self.logger.add_scalar(f"valid/{k}", m.value, self.num_epochs)
            print(f"epoch {epoch:02d}",
                  ", ".join(f"{k}: {v}" for k, v in metrics.items()),
                  f"(avg train loss: {train_loss:.5e})")
            auc_values = [v.value for k, v in metrics.items() if k.startswith('auc')]
            if len(auc_values) > 1:
                avg_auc = 100 * sum(auc_values) / len(auc_values)
                self.logger.add_scalar(f"valid/avg_auc", avg_auc, self.num_epochs)
                print(f"avg AUC: {avg_auc:05.2f}%")

        return best_metrics
