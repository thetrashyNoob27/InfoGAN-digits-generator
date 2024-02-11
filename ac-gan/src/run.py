import yaml
import argparse
import tensorflow as tf

from utils import load_data
import simple_train
import cond_train
import ac_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    with open(_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(cfg)
    print(tf.config.list_physical_devices('GPU'))

    train_ds = load_data(cfg)

    if cfg['model'] == 'simple_gan':
        simple_train.train(cfg, train_ds)
    elif cfg['model'] == 'cond_gan':
        cond_train.train(cfg, train_ds)
    elif cfg['model'] == 'ac_gan':
        ac_train.train(cfg, train_ds)
    else:
        raise NotImplementedError()