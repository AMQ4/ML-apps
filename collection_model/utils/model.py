import cloudpickle
from config import MODEL_PATH

def load_model():
    """
    Load a pre-trained model from a file using cloudpickle.

    The model is loaded from the path specified in the MODEL_PATH variable,
    which is defined in the config module.

    Returns:
        model: The loaded model object.
    """
    with open(MODEL_PATH, 'rb') as f:
        model = cloudpickle.load(f)
    return model
