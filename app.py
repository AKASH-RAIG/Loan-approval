# Minimal Loan Prediction System with Frontend Integration (No Graphs)

import pandas as pd
import joblib
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load dataset
df = pd.read_csv("loan_data_set_augmented(1).csv")

# Define columns
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Fill missing values
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Encode categorical features
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save encoders and scaler
joblib.dump(encoders, "encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

# Model training
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

# Load saved models
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        'Gender': request.form['Gender'],
        'Married': request.form['Married'],
        'Dependents': request.form['Dependents'],
        'Education': request.form['Education'],
        'Self_Employed': request.form['Self_Employed'],
        'ApplicantIncome': float(request.form['ApplicantIncome']),
        'CoapplicantIncome': float(request.form['CoapplicantIncome']),
        'LoanAmount': float(request.form['LoanAmount']),
        'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
        'Credit_History': float(request.form['Credit_History']),
        'Property_Area': request.form['Property_Area']
    }

    user_df = pd.DataFrame([user_input])

    for col in cat_cols:
        if user_df[col].iloc[0] not in encoders[col].classes_:
            user_df[col] = encoders[col].transform([encoders[col].classes_[0]])
        else:
            user_df[col] = encoders[col].transform(user_df[col])

    user_df[num_cols] = scaler.transform(user_df[num_cols])

    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[X.columns]

    prediction = model.predict(user_df)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"

    return render_template('index.html', prediction_text=f'Loan Prediction Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)