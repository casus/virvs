from dataclasses import dataclass


@dataclass
class EvalConfig:
    output_path: str
    log_freq: int
    val_freq: int
