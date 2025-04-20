import yaml
from virvs.configs.data_config import DataConfig
from virvs.configs.eval_config import EvalConfig
from virvs.configs.neptune_config import NeptuneConfig
from virvs.configs.training_config import TrainingConfig


def load_config_from_yaml(yaml_file):
    """
    Load configuration data from a YAML file.

    This function reads a YAML file, parses its contents, and returns the data 
    as a Python dictionary.

    Args:
        yaml_file (str): The path to the YAML file to load.

    Returns:
        dict: A dictionary containing the configuration data loaded from the YAML file.
    """
    with open(yaml_file, "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def create_training_config(config_data):
    """
    Create a `TrainingConfig` object from the configuration data.

    This function extracts the relevant fields for training configuration from 
    the provided configuration data and initializes a `TrainingConfig` object.

    Args:
        config_data (dict): A dictionary containing the configuration data. 
                             Specifically, it should have a "training" key.

    Returns:
        TrainingConfig: An instance of the `TrainingConfig` class initialized 
                        with values from the configuration data.
    """
    return TrainingConfig(**config_data["training"])


def create_data_config(config_data):
    """
    Create a `DataConfig` object from the configuration data.

    This function extracts the relevant fields for data configuration from 
    the provided configuration data and initializes a `DataConfig` object.

    Args:
        config_data (dict): A dictionary containing the configuration data. 
                             Specifically, it should have a "data" key.

    Returns:
        DataConfig: An instance of the `DataConfig` class initialized with values 
                    from the configuration data.
    """
    return DataConfig(**config_data["data"])


def create_eval_config(config_data):
    """
    Create an `EvalConfig` object from the configuration data.

    This function extracts the relevant fields for evaluation configuration from 
    the provided configuration data and initializes an `EvalConfig` object.

    Args:
        config_data (dict): A dictionary containing the configuration data. 
                             Specifically, it should have an "eval" key.

    Returns:
        EvalConfig: An instance of the `EvalConfig` class initialized with values 
                    from the configuration data.
    """
    return EvalConfig(**config_data["eval"])


def create_neptune_config(config_data):
    """
    Create a `NeptuneConfig` object from the configuration data.

    This function checks if "neptune" configuration exists in the provided data 
    and, if so, creates and returns an instance of `NeptuneConfig`.

    Args:
        config_data (dict): A dictionary containing the configuration data. 
                             Optionally, it can have a "neptune" key.

    Returns:
        NeptuneConfig or None: An instance of the `NeptuneConfig` class if 
                                the "neptune" key exists, otherwise `None`.
    """
    if "neptune" in config_data:
        return NeptuneConfig(**config_data["neptune"])
    else:
        return None
