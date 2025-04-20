from dataclasses import dataclass


@dataclass
class NeptuneConfig:
    """
    A configuration class to hold Neptune-related settings.
    
    This class stores configuration settings required to integrate with the Neptune logging 
    platform, such as the project name and the specific run name for tracking the experiment.

    Attributes:
        name (str): The name of the Neptune run to track.
        project (str): The Neptune project where the experiment will be logged.
    """
    name: str
    project: str
