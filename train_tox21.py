import time

from tox21 import train

if __name__ == "__main__":
    from upsilonconf import load_config

    config = load_config("config")
    hparams, sys_config = config.hparams, config.system
    log_dir = time.strftime("runs/tox21/%y%j-%H%M%S")
    train.run(hparams, sys_config, log_dir)
