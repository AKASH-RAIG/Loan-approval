@app.route('/predict', methods=['POST'])
def predict():
    # Get form values and store original income and loa values before scaling
    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])

    user_input = {
        'Gender': request.form['Gender'],
        'Married': request.form['Married'],
        'Dependents': request.form['Dependents'],
        'Education': request.form['Education'],
        'Self_Employed': request.form['Self_Employed'],
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
        'Credit_History': float(request.form['Credit_History']),
        'Property_Area': request.form['Property_Area']
    }

    # Derived features
    income_total = applicant_income + coapplicant_income
    income_loan_ratio = income_total / (loan_amount + 1)

    user_input['Income_Total'] = income_total
    user_input['Income_Loan_Ratio'] = income_loan_ratio

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

    # Rule-based override (comparison on unscaled values)
    if income_total > loan_amount:
        result = "✅ Approved (Rule-based: Income > Loan Amount)"
    else:
        prediction = model.predict(user_df)[0]
        result = "✅ Approved" if prediction == 1 else "❌ Rejected"

    return render_template('index.html', prediction_text=f'Loan Prediction Result: {result}')
