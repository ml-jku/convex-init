import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
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
    for layer in _iterate_layers(model):
        x = layer(*x)

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
        self.model = model
        self.objective = objective
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
        for batch in batches:
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
        for batch in batches:
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

        for epoch in range(1, num_epochs + 1):
            train_loss = self.update(train_loader)
            metrics = self.evaluate(valid_loader, extra_metrics)
            if self.logger is not None:
                self.logger.add_scalar("train/avg_loss", train_loss, self.num_epochs)
                for k, m in metrics.items():
                    self.logger.add_scalar(f"valid/{k}", m.value, self.num_epochs)
            print(f"epoch {epoch:02d}",
                  ", ".join(f"{k}: {v}" for k, v in metrics.items()),
                  f"(avg train loss: {train_loss:.5e})")

        return metrics
