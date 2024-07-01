from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import logging
from pymongo import MongoClient
from bson.objectid import ObjectId

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['diabetes_prediction_db']
collection = db['predictions']

# Use an absolute path for the model
model_path = os.path.join(os.path.dirname(__file__), 'diabetes_prediction_model.joblib')
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logging.debug(f"Received data: {data}")

        # Verify the data structure
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            logging.debug(f"DataFrame: {df}")

            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)
            logging.debug(f"Prediction: {prediction}")
            logging.debug(f"Prediction Probability: {prediction_proba}")

            # Store the input data and prediction result in MongoDB
            for i in range(len(data)):
                record = data[i]
                record['prediction'] = int(prediction[i])
                record['confidence'] = float(max(prediction_proba[i]))
                collection.insert_one(record)

            return jsonify({"prediction": prediction.tolist(), "confidence": prediction_proba.tolist()})
        else:
            logging.error("Invalid data format")
            return jsonify({"error": "Invalid data format"}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/data', methods=['GET'])
def get_user_data():
    try:
        # Fetch all records from the MongoDB collection
        records = list(collection.find({}, {'_id': 1, 'gender': 1, 'age': 1, 'hypertension': 1, 'heart_disease': 1, 'smoking_history': 1, 'bmi': 1, 'HbA1c_level': 1, 'blood_glucose_level': 1, 'prediction': 1, 'confidence': 1}))  # Include MongoDB internal ID field for deletion
        for record in records:
            record['_id'] = str(record['_id'])  # Convert ObjectId to string for easier handling in frontend
        return jsonify(records)
    except Exception as e:
        logging.error(f"Error fetching user data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/data/<id>', methods=['DELETE'])
def delete_user_data(id):
    try:
        result = collection.delete_one({'_id': ObjectId(id)})
        if result.deleted_count == 1:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logging.error(f"Error deleting user data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
