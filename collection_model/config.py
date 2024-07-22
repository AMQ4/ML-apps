import os

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the preprocessing pipeline file
PIPELINE_PATH = os.path.join(BASE_DIR, 'utils', 'preprocessing_pipeline.pkl')

# Path to the model file
MODEL_PATH = os.path.join(BASE_DIR, 'utils', 'model.pkl')
