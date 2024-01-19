import upsilonconf

common_space = {
    "pre_process": ["normal", "pca", "zca"],
    "adam.lr": [1e-2, 1e-3, 1e-4],
    "adam.weight_decay": [0., 1e-2],
    "model.hidden": [
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
}

cifar_space = {"data.name": ["CIFAR10"]} | common_space
mnist_space = {
    "data.name": ["MNIST"],
    # original idea was to have different search spaces, but copy-paste bug
    # "model.hidden": [
    #     (100, ),
    #     (1000, ),
    #     (100, 100),
    #     (100, 1000),
    #     (1000, 100),
    #     (1000, 1000),
    #     (100, 100, 100),
    #     (100, 1000, 100),
    #     (1000, 100, 1000),
    #     (1000, 1000, 1000),
    # ]
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

    datasets = {
        "mnist": mnist_space,
        "cifar": cifar_space,
    }

    models = {
        "nn": {"convex": "", "fix_init": True, "skip": False},
        "icnn": {"convex": "icnn", "fix_init": False, "skip": False},
        "skip": {"convex": "icnn", "fix_init": False, "skip": True},
        "ours": {"convex": "icnn", "fix_init": True, "skip": False},
        "exp1": {"convex": "exp", "fix_init": False, "skip": False},
        "exp2": {"convex": "exp", "fix_init": True, "skip": False}
    }

    for model_prefix, model_conf in models.items():
        for data_prefix, search_space in datasets.items():
            config_subdir = config_dir / model_prefix / data_prefix
            config_subdir.mkdir(exist_ok=False, parents=True)
            config = upsilonconf.Configuration(**base_config)
            config.overwrite("model", config.model | model_conf)
            for i, conf in enumerate(cartesian_dict(dict(search_space)), 18):
                config.overwrite("id", f"{model_prefix}_{data_prefix}{i:03d}")
                config.overwrite_all(conf)
                upsilonconf.save(config, config_subdir / f"config{i:03d}.yaml")
