import upsilonconf

common_space = {
    "pre_process": ["normal", "pca", "zca"],
    "adam.lr": [1e-2, 1e-3, 1e-4],
    "adam.weight_decay": [0., 1e-2],
}

mnist_space = {
    "model.hidden": [
        (),
        (100, ),
        (1000, ),
        (100, 100),
        (100, 1000),
        (1000, 100),
        (1000, 1000),
        (100, 100, 100),
        (100, 1000, 100),
        (1000, 100, 1000),
        (1000, 1000, 1000),
    ]
} | common_space

cifar_space = {
    "model.hidden": [
        (),
        (1000, ),
        (10000, ),
        (1000, 1000),
        (1000, 10000),
        (10000, 1000),
        (10000, 10000),
        (1000, 1000, 1000),
        (1000, 10000, 1000),
        (10000, 1000, 10000),
        (10000, 10000, 10000),
    ]
} | common_space


def cartesian_dict(d: dict):
    if len(d) == 0:
        yield {}
        return

    key, values = d.popitem()
    for sub in cartesian_dict(d):
        for val in values:
            yield dict(**sub, **{key: val})


if __name__ == "__main__":
    from pathlib import Path
    config_dir = Path("options")
    config_dir.mkdir(exist_ok=True)
    base_config = upsilonconf.load("default.yaml")

    cifar_config_dir = config_dir / "cifar"
    cifar_config_dir.mkdir(exist_ok=False)
    cifar_config = upsilonconf.Configuration(**base_config)
    cifar_config.overwrite("data.name", "CIFAR10")
    for i, conf in enumerate(cartesian_dict(dict(cifar_space))):
        cifar_config.overwrite("id", f"cifar{i:03d}")
        cifar_config.overwrite_all(conf)
        upsilonconf.save(cifar_config, cifar_config_dir / f"config{i:03d}.yaml")

    mnist_config_dir = config_dir / "mnist"
    mnist_config_dir.mkdir(exist_ok=False)
    mnist_config = upsilonconf.Configuration(**base_config)
    mnist_config.overwrite("data.name", "MNIST")
    for i, conf in enumerate(cartesian_dict(dict(cifar_space))):
        mnist_config.overwrite("id", f"mnist{i:03d}")
        mnist_config.overwrite_all(conf)
        upsilonconf.save(mnist_config, mnist_config_dir / f"config{i:03d}.yaml")
