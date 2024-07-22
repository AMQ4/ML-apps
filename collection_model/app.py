import pandas as pd
from flask import Flask, render_template, request
from utils.preprocessing import load_pipeline
from utils.model import load_model

app = Flask(__name__)
df = None

@app.route('/')
def index():
    """
    Render the index.html template.

    Returns:
        Rendered HTML template for the index page.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload from the user. Reads a CSV file and stores it in a global DataFrame.

    Returns:
        str: Message indicating the result of the file upload process.
    """
    global df
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        return f"File {file.filename} uploaded successfully"
    
    return "Invalid file format"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Perform predictions on the uploaded CSV file using a pre-trained model and a preprocessing pipeline.

    Returns:
        str: Message indicating the predicted collections for the next month.
    """
    global df
    if df is None:
        return "No file uploaded"

    pipeline = load_pipeline()
    model = load_model()

    transformed_df = pipeline.transform(df)
    predictions = model.predict(transformed_df)

    return f"Predicted Collections for next month: {round(predictions.sum(), 2)} JOD."

if __name__ == '__main__':
    """
    Run the Flask application in debug mode.
    """
    app.run(debug=True)
