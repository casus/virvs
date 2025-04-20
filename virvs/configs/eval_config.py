from dataclasses import dataclass


@dataclass
class EvalConfig:
    """
    A configuration class to hold evaluation-related settings.
    
    This class stores settings for the evaluation process, including the output path for saving results,
    the frequency of logging during training, and the frequency of validation steps. These settings are 
    important for controlling how often and where evaluation data is saved or logged.

    Attributes:
        output_path (str): The directory path where evaluation results and outputs will be saved.
        log_freq (int): The frequency (in steps) at which the training loss and metrics should be logged.
        val_freq (int): The frequency (in steps) at which the model will be evaluated on the validation set.
    """
    output_path: str
    log_freq: int
    val_freq: int
