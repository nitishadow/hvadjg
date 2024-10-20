from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from model.lifestyle_model import predict_lifestyle_score

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Allow localhost:63343

# POST API to predict lifestyle score
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Check if all required fields are present
        required_fields = [
            'Age', 'Daily_Steps', 'Calories_Consumed', 'Sleep_Hours',
            'Water_Intake_Liters', 'Exercise_Hours', 'BMI', 'Gender_Male',
            'Stress_Level_Medium', 'Stress_Level_High'
        ]
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields in the input data'}), 400

        # Predict lifestyle score
        score = predict_lifestyle_score(data)
        return jsonify({'Healthy_Lifestyle_Score': round(score, 2)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
