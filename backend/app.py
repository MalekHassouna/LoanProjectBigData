import pandas as pd
import joblib
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS  # Permettre les requêtes CORS
import os
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)
CORS(app)

# Charger les modèles
models = {
    "logistic_regression": joblib.load(r"C:\Users\MSI\Desktop\Big Data\loanBigData\backend\models\Logistic_Regression_model.pkl"),
    "random_forest": joblib.load(r"C:\Users\MSI\Desktop\Big Data\loanBigData\backend\models\Random_Forest_model.pkl"),
    "xgboost": joblib.load(r"C:\Users\MSI\Desktop\Big Data\loanBigData\backend\models\XGBoost_model.pkl"),
    "catboost": joblib.load(r"C:\Users\MSI\Desktop\Big Data\loanBigData\backend\models\CatBoost_model.pkl"),
}

# Colonnes requises
required_columns = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
    'previous_loan_defaults_on_file', 'person_home_ownership_OTHER',
    'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]

# Fonction de prétraitement
def preprocess_new_data(data):
    # Binary Encoding
    data['person_gender'] = data['person_gender'].map({'female': 0, 'male': 1})
    data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

    # Ordinal Encoding pour 'person_education'
    education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
    data['person_education'] = data['person_education'].map(education_order)

    # One-Hot Encoding pour 'person_home_ownership' et 'loan_intent'
    data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

    # Ajout des colonnes manquantes avec des valeurs par défaut
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0

    # Réorganiser les colonnes selon 'required_columns'
    data = data[required_columns]
    return data

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    df = pd.read_csv(file)

    # Vérifier le modèle choisi
    model_choice = request.form.get('model')
    if model_choice not in models:
        return "Invalid model selected", 400

    model = models[model_choice]

    try:
        # Prétraiter les données
        processed_data = preprocess_new_data(df)

        # Normalisation des données avec RobustScaler
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(processed_data)

        # Prédire les résultats
        predictions = model.predict(scaled_data)
        df['Prediction'] = predictions

        # Enregistrer les résultats dans un fichier CSV
        output_file = 'output_predictions.csv'
        df.to_csv(output_file, index=False)

        # Répondre avec le fichier
        return send_file(output_file, as_attachment=True, download_name='output_predictions.csv')

    except Exception as e:
        return f"Prediction failed: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
