from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """
    A configuration class to hold data-related settings.
    
    This class stores paths for training and validation data, batch size, image size, 
    and the input channels for the dataset. It is used to define the necessary 
    configuration for loading and processing the dataset during training and evaluation.

    Attributes:
        train_data_path (str): The path to the training dataset.
        val_data_path (str): The path to the validation dataset.
        batch_size (int): The batch size to use for training (for validation batch of 1 is used).
        im_size (int): The size (height/width) of the images in the dataset. Assumes square images.
        ch_in (List[int]): A list containing the idxs of the input channels.
    """
    train_data_path: str
    val_data_path: str
    batch_size: int
    im_size: int
    ch_in: List[int]
