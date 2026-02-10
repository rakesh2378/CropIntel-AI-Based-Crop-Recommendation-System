import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import pandas as pd
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/crop_standards.db'
db = SQLAlchemy(app)

# Load the trained CNN model for soil type prediction
cnn_model = load_model('models/soil_cnn_model.h5')
soil_types = ['Black Soil', 'Red soil', 'Alluvial soil', 'Clay soil']

# Load the trained tabular model for crop recommendation
tabular_model = joblib.load('models/tabular_crop_model.pkl')

# Soil types and their recommended crops (for image model)
SOIL_CROP_MAP = {
    'Black Soil': ['Cotton', 'Soybean', 'Wheat'],
    'Red soil': ['Millets', 'Groundnut', 'Pulses'],
    'Alluvial soil': ['Rice', 'Sugarcane', 'Wheat'],
    'Clay soil': ['Rice', 'Jute', 'Sugarcane']
}

# Add this mapping at the top of your file
COLUMN_MAP = {
    'Soil pH': 'ph',
    'Salinity': 'salinity',
    'Moisture': 'moisture',
    'Acidity': 'acidity',
    'Fertility rate': 'fertility',
    'Water availability': 'water',
    'Wind': 'wind',
    'Sunlight': 'sunlight',
    'Temperature': 'temperature',
    'Rainfall': 'rainfall',
    'Photosynthesis activity': 'photosynthesis',
    'Microbial content': 'microbial',
    'Pollination agents presence': 'pollination',
    # Add more mappings as needed
}

POLLINATION_MAP = {
    'None': 0,
    'Bees': 1,
    'Butterflies': 2,
    'Wind': 3
    # Add more as needed
}

def predict_soil_type(img_file):
    img_bytes = img_file.read()  # Read the file bytes
    img_file.seek(0)  # Reset file pointer for future use
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = cnn_model.predict(x)
    predicted_soil = soil_types[np.argmax(preds)]
    return predicted_soil

def predict_crop_from_tabular(features):
    pred = tabular_model.predict([features])
    return pred[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data_file = request.files.get('file')
        crops_from_data = []
        if data_file and (data_file.filename.lower().endswith('.csv') or data_file.filename.lower().endswith('.xlsx')):
            if data_file.filename.lower().endswith('.csv'):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)
            df = df.rename(columns=COLUMN_MAP)
            if 'pollination' in df.columns:
                df['pollination'] = df['pollination'].apply(lambda x: POLLINATION_MAP.get(x, x) if isinstance(x, str) and not str(x).isdigit() else x)
            features = df.iloc[0][['ph','salinity','moisture','acidity','fertility','water','wind','sunlight','temperature','rainfall','photosynthesis','microbial','pollination']].tolist()
            predicted_crop = predict_crop_from_tabular(features)
            crops_from_data = [{'name': predicted_crop, 'score': 99}]
        session['crops'] = crops_from_data
        flash('File uploaded and processed successfully!', 'success')
        return redirect(url_for('results'))
    return render_template('upload.html')

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and (file.filename.lower().endswith('.jpg') or file.filename.lower().endswith('.jpeg') or file.filename.lower().endswith('.png')):
            predicted_soil = predict_soil_type(file)
            crops = SOIL_CROP_MAP.get(predicted_soil, [])
            session['soil_type'] = predicted_soil
            session['crops'] = [
                {'name': crop, 'score': random.randint(80, 99)} for crop in crops
            ]
            flash(f'Soil type detected: {predicted_soil}', 'success')
            return redirect(url_for('results'))
        else:
            flash('Please upload a valid JPG, JPEG, or PNG image.', 'danger')
    return render_template('upload_image.html')

@app.route('/manual-entry', methods=['GET', 'POST'])
def manual_entry():
    if request.method == 'POST':
        # Extract features from form in the same order as training
        features = [
            float(request.form.get('ph', 0)),
            float(request.form.get('salinity', 0)),
            float(request.form.get('moisture', 0)),
            float(request.form.get('acidity', 0)),
            float(request.form.get('fertility', 0)),
            float(request.form.get('water', 0)),
            float(request.form.get('wind', 0)),
            float(request.form.get('sunlight', 0)),
            float(request.form.get('temperature', 0)),
            float(request.form.get('rainfall', 0)),
            float(request.form.get('photosynthesis', 0)),
            float(request.form.get('microbial', 0)),
            float(request.form.get('pollination', 0))
        ]
        pollination_value = request.form.get('pollination', 0)
        if isinstance(pollination_value, str) and not pollination_value.isdigit():
            pollination_value = POLLINATION_MAP.get(pollination_value, 0)
        features[12] = float(pollination_value)
        predicted_crop = predict_crop_from_tabular(features)
        session['crops'] = [{'name': predicted_crop, 'score': 99}]
        flash('Data submitted and processed successfully!', 'success')
        return redirect(url_for('results'))
    return render_template('manual_entry.html')

@app.route('/results')
def results():
    crops = session.get('crops', [])
    soil_type = session.get('soil_type', None)
    return render_template('results.html', crops=crops, soil_type=soil_type)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        flash('Admin upload/update (mock) successful!', 'success')
    return render_template('admin.html')

@app.route('/combined-upload', methods=['GET', 'POST'])
def combined_upload():
    if request.method == 'POST':
        image_file = request.files.get('image')
        data_file = request.files.get('datafile')
        crops_from_image = []
        crops_from_data = []
        crops_from_manual = []
        soil_type = None
        # Image logic
        if image_file and (image_file.filename.lower().endswith('.jpg') or image_file.filename.lower().endswith('.jpeg') or image_file.filename.lower().endswith('.png')):
            soil_type = predict_soil_type(image_file)
            crops_from_image = SOIL_CROP_MAP.get(soil_type, [])
        # Dataset logic
        if data_file and (data_file.filename.lower().endswith('.csv') or data_file.filename.lower().endswith('.xlsx')):
            if data_file.filename.lower().endswith('.csv'):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)
            df = df.rename(columns=COLUMN_MAP)
            if 'pollination' in df.columns:
                df['pollination'] = df['pollination'].apply(lambda x: POLLINATION_MAP.get(x, x) if isinstance(x, str) and not str(x).isdigit() else x)
            features = df.iloc[0][['ph','salinity','moisture','acidity','fertility','water','wind','sunlight','temperature','rainfall','photosynthesis','microbial','pollination']].tolist()
            predicted_crop = predict_crop_from_tabular(features)
            crops_from_data = [{'name': predicted_crop, 'score': 99}]
        # Manual entry logic
        if not crops_from_data:
            features = [
                float(request.form.get('ph', 0)),
                float(request.form.get('salinity', 0)),
                float(request.form.get('moisture', 0)),
                float(request.form.get('acidity', 0)),
                float(request.form.get('fertility', 0)),
                float(request.form.get('water', 0)),
                float(request.form.get('wind', 0)),
                float(request.form.get('sunlight', 0)),
                float(request.form.get('temperature', 0)),
                float(request.form.get('rainfall', 0)),
                float(request.form.get('photosynthesis', 0)),
                float(request.form.get('microbial', 0)),
                float(request.form.get('pollination', 0))
            ]
            pollination_value = request.form.get('pollination', 0)
            if isinstance(pollination_value, str) and not pollination_value.isdigit():
                pollination_value = POLLINATION_MAP.get(pollination_value, 0)
            features[12] = float(pollination_value)
            if any(features):
                predicted_crop = predict_crop_from_tabular(features)
                crops_from_manual = [{'name': predicted_crop, 'score': 99}]
        # Combine crops (priority: image > dataset > manual)
        crops = []
        if crops_from_image:
            crops = [
                {'name': crop, 'score': random.randint(80, 99)} for crop in crops_from_image
            ]
        elif crops_from_data:
            crops = crops_from_data
        elif crops_from_manual:
            crops = crops_from_manual
        session['soil_type'] = soil_type
        session['crops'] = crops
        flash('Files/data submitted and processed successfully!', 'success')
        return redirect(url_for('results'))
    return render_template('combined_upload.html')

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True) 
