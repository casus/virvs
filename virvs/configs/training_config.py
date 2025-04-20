from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    A configuration class to hold training-related settings.

    This class stores parameters related to the training process, such as the maximum number 
    of training steps and the weight assigned to the discriminator in the Pix2Pix model.

    Attributes:
        max_steps (int): The maximum number of training steps to run the model.
        pix2pix_disc_weight (float): The weight factor applied to the discriminator loss in 
                                      the Pix2Pix model.
    """
    max_steps: int
    pix2pix_disc_weight: float
