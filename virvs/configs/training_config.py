from dataclasses import dataclass


@dataclass
class TrainingConfig:
    max_steps: int
    pix2pix_disc_weight: float
