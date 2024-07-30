import yaml
from virvs.configs.data_config import DataConfig
from virvs.configs.eval_config import EvalConfig
from virvs.configs.neptune_config import NeptuneConfig
from virvs.configs.training_config import TrainingConfig


def load_config_from_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def create_training_config(config_data):
    return TrainingConfig(**config_data["training"])


def create_data_config(config_data):
    return DataConfig(**config_data["data"])


def create_eval_config(config_data):
    return EvalConfig(**config_data["eval"])


def create_neptune_config(config_data):
    if "neptune" in config_data:
        return NeptuneConfig(**config_data["neptune"])
    else:
        return None
