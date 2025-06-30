# app.py (Corrected to match the final training script)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('fraud_model.joblib')
scaler = joblib.load('data_scaler.joblib')

# This is our "source of truth" for column order
model_columns = ['scaled_amount', 'scaled_time'] + \
    [f'V{i}' for i in range(1, 29)]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        df_live = pd.DataFrame(data, index=[0])

        # --- Preprocessing: Must be identical to training ---
        cols_to_scale = ['Amount', 'Time']

        # 1. Check for required columns
        if not all(col in df_live.columns for col in cols_to_scale):
            return jsonify({'error': 'Missing Time or Amount in input data'}), 400

        # 2. Transform the two columns together and add them as new columns
        df_live[['scaled_amount', 'scaled_time']
                ] = scaler.transform(df_live[cols_to_scale])

        # 3. Drop the original columns
        df_live = df_live.drop(cols_to_scale, axis=1)

        # 4. Enforce the exact final column order
        df_live = df_live[model_columns]

    except Exception as e:
        return jsonify({'error': f'Error during data processing: {str(e)}'}), 400

    # --- Prediction ---
    prediction = model.predict(df_live)
    probability = model.predict_proba(df_live)

    # --- Response ---
    is_fraud = bool(prediction[0])
    confidence = probability[0][1] if is_fraud else probability[0][0]

    return jsonify({
        'is_fraud': is_fraud,
        'fraud_probability': f'{probability[0][1]:.4f}',
        'confidence': f'{confidence:.2%}',
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
