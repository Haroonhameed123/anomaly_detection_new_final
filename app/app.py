from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import json

app = Flask(__name__)

# LSTM Model Classes (same as your training code)
class ConservativeOptimizedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=3, dropout_rate=0.25):
        super(ConservativeOptimizedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate,
            bidirectional=False
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.batch_norm(lstm_out)
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output

class SlightlyImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=112, num_layers=3, dropout_rate=0.22):
        super(SlightlyImprovedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate,
            bidirectional=False
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.batch_norm(lstm_out)
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output

# Global variables for model and scaler
model = None
scaler = None
model_info = {}

def load_model_and_scaler():
    """Load the best model and create/load scaler"""
    global model, scaler, model_info
    
    try:
        # Try to load the best model (adjust path as needed)
        model_path = "../models/best_SlightlyImprovedLSTMModel_20250623_224633.pth"
        
        if not os.path.exists(model_path):
            # If model doesn't exist, create a dummy one for demo
            print("Model file not found. Creating dummy model for demo...")
            model = SlightlyImprovedLSTMModel(input_size=7)
            model.eval()
            model_info = {
                'model_type': 'SlightlyImprovedLSTMModel (Demo)',
                'status': 'Demo mode - using random weights'
            }
        else:
            # Load the actual trained model
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['model_config']
            
            if config['model_class'] == 'SlightlyImprovedLSTMModel':
                model = SlightlyImprovedLSTMModel(
                    input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout_rate=config['dropout_rate']
                )
            else:
                model = ConservativeOptimizedLSTMModel(
                    input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout_rate=config['dropout_rate']
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            model_info = {
                'model_type': config['model_class'],
                'validation_loss': checkpoint['best_val_loss'],
                'epochs_trained': checkpoint['epoch'] + 1,
                'status': 'Loaded successfully'
            }
        
        # Create scaler with approximate ranges from your data
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit with approximate min/max values from your dataset
        dummy_data = np.array([
            [0.076, 0.0, 223.2, 0.2, 0.0, 0.0, 0.0],  # min values
            [11.122, 1.39, 254.15, 48.4, 88.0, 80.0, 31.0]  # max values
        ])
        scaler.fit(dummy_data)
        
        print(f"Model loaded: {model_info}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create dummy model as fallback
        model = SlightlyImprovedLSTMModel(input_size=7)
        model.eval()
        scaler = MinMaxScaler(feature_range=(0, 1))
        dummy_data = np.array([
            [0.076, 0.0, 223.2, 0.2, 0.0, 0.0, 0.0],
            [11.122, 1.39, 254.15, 48.4, 88.0, 80.0, 31.0]
        ])
        scaler.fit(dummy_data)
        model_info = {'status': 'Demo mode - error loading trained model'}

def predict_power_consumption(input_data):
    """Make prediction using the loaded model"""
    global model, scaler
    
    try:
        # Convert input to numpy array
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input
        scaled_input = scaler.transform(input_array)
        
        # Reshape for LSTM (batch_size, sequence_length, features)
        lstm_input = scaled_input.reshape(1, 1, -1)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(lstm_input)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()
        
        # Inverse transform prediction
        # Create dummy array for inverse transform
        dummy_array = np.zeros((1, 7))
        dummy_array[0, 0] = prediction[0, 0]
        inverse_scaled = scaler.inverse_transform(dummy_array)
        final_prediction = inverse_scaled[0, 0]
        
        return float(final_prediction)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        
        # Extract input features
        features = [
            float(data.get('global_active_power', 1.0)),
            float(data.get('global_reactive_power', 0.1)),
            float(data.get('voltage', 240.0)),
            float(data.get('global_intensity', 4.0)),
            float(data.get('sub_metering_1', 1.0)),
            float(data.get('sub_metering_2', 1.0)),
            float(data.get('sub_metering_3', 6.0))
        ]
        
        # Make prediction
        prediction = predict_power_consumption(features)
        
        if prediction is not None:
            return jsonify({
                'success': True,
                'prediction': round(prediction, 4),
                'input_features': {
                    'global_active_power': features[0],
                    'global_reactive_power': features[1],
                    'voltage': features[2],
                    'global_intensity': features[3],
                    'sub_metering_1': features[4],
                    'sub_metering_2': features[5],
                    'sub_metering_3': features[6]
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model_info')
def get_model_info():
    """Get information about the loaded model"""
    return jsonify(model_info)

@app.route('/generate_sample', methods=['GET'])
def generate_sample():
    """Generate sample input values based on dataset statistics"""
    # Sample values based on your dataset statistics
    samples = {
        'low_consumption': {
            'global_active_power': 0.31,
            'global_reactive_power': 0.048,
            'voltage': 239.02,
            'global_intensity': 1.4,
            'sub_metering_1': 0.0,
            'sub_metering_2': 0.0,
            'sub_metering_3': 0.0
        },
        'medium_consumption': {
            'global_active_power': 1.52,
            'global_reactive_power': 0.192,
            'voltage': 242.86,
            'global_intensity': 6.4,
            'sub_metering_1': 1.0,
            'sub_metering_2': 1.0,
            'sub_metering_3': 17.0
        },
        'high_consumption': {
            'global_active_power': 4.5,
            'global_reactive_power': 0.4,
            'voltage': 245.0,
            'global_intensity': 18.0,
            'sub_metering_1': 10.0,
            'sub_metering_2': 8.0,
            'sub_metering_3': 25.0
        }
    }
    
    return jsonify(samples)

if __name__ == '__main__':
    # Load model and scaler on startup
    load_model_and_scaler()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)