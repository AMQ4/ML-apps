import cloudpickle
from config import PIPELINE_PATH

def load_pipeline():
    """
    Load a preprocessing pipeline from a file using cloudpickle.

    The pipeline is loaded from the path specified in the PIPELINE_PATH variable,
    which is defined in the config module.

    Returns:
        full_pipeline: The loaded preprocessing pipeline object.
    """
    with open(PIPELINE_PATH, 'rb') as f:
        full_pipeline = cloudpickle.load(f)

    return full_pipeline
