from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    train_data_path: str
    val_data_path: str
    batch_size: int
    im_size: int
    ch_in: List[int]
