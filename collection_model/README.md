# Debt Collection Prediction Flask App

This is a Flask web application for predicting debt collections for the next month using a pre-trained machine learning model and a preprocessing pipeline.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Configuration](#configuration)
- [License](#license)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/debt-collection-prediction.git
    cd debt-collection-prediction
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Place your pre-trained model and preprocessing pipeline files in the `utils` directory:**
    - `preprocessing_pipeline.pkl`
    - `model.pkl`

## Usage

1. **Run the Flask application:**
    ```bash
    python app.py
    ```

2. **Open your web browser and go to:**
    ```
    http://127.0.0.1:5000/
    ```

3. **Upload a CSV file:**
    - Use the provided form to upload your CSV file.

4. **Get Predictions:**
    - Click on the `Predict` button to get the predicted debt collections for the next month.

## File Descriptions

- **app.py:** Main Flask application file.
- **config.py:** Configuration file for specifying paths to model and pipeline files.
- **model.py:** Contains the function to load the pre-trained model.
- **preprocessing.py:** Contains the function to load the preprocessing pipeline.
- **requirements.txt:** Lists the dependencies needed to run the application.

## Configuration

The configuration is handled in `config.py`. Ensure that the paths to your model and preprocessing pipeline files are correct:

```python
import os

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the preprocessing pipeline file
PIPELINE_PATH = os.path.join(BASE_DIR, 'utils', 'preprocessing_pipeline.pkl')

# Path to the model file
MODEL_PATH = os.path.join(BASE_DIR, 'utils', 'model.pkl')
