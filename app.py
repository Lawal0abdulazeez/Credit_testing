from flask import Flask, request, render_template_string, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import sys
from CC import MultiColumnLabelEncoder  # Ensure this imports your MultiColumnLabelEncoder
sys.modules['__main__'].MultiColumnLabelEncoder = MultiColumnLabelEncoder

# Load the preprocessor and credit scoring model.
credit_preprocessor = joblib.load("preprocessor_Credit Score.pkl")
credit_model = joblib.load("Lasso_Credit Score.pkl")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flashing messages

# -------------------------------
# HTML Templates
# -------------------------------
home_template = """
<!doctype html>
<html lang="en">
  <head>
    <title>Credit Score Calculator</title>
  </head>
  <body>
    <h1>Enter Your Details</h1>
    <form action="{{ url_for('predict_credit_score_ui') }}" method="post">
      <label for="Age">Age:</label><br>
      <input type="number" id="Age" name="Age" value="30" required><br><br>

      <label for="Gender">Gender:</label><br>
      <input type="text" id="Gender" name="Gender" value="Male" required><br><br>

      <label for="Risk_Ratings">Risk Ratings:</label><br>
      <input type="text" id="Risk_Ratings" name="Risk_Ratings" value="Medium" required><br><br>

      <label for="Location">Location:</label><br>
      <input type="text" id="Location" name="Location" value="Lagos" required><br><br>

      <label for="Delinquency_Frequency">Delinquency Frequency:</label><br>
      <input type="number" id="Delinquency_Frequency" name="Delinquency_Frequency" value="3" required><br><br>

      <label for="Marital_Status">Marital Status:</label><br>
      <input type="text" id="Marital_Status" name="Marital_Status" value="Single" required><br><br>

      <label for="Monthly_Income">Monthly Income (NGN):</label><br>
      <input type="number" id="Monthly_Income" name="Monthly_Income" value="500000" required><br><br>

      <label for="Debt_to_Income_Ratio">Debt-to-Income Ratio:</label><br>
      <input type="number" step="any" id="Debt_to_Income_Ratio" name="Debt_to_Income_Ratio" value="0.5" required><br><br>

      <label for="Working_Sector">Working Sector:</label><br>
      <input type="text" id="Working_Sector" name="Working_Sector" value="Finance" required><br><br>

      <label for="Loan_Types">Loan Types:</label><br>
      <input type="text" id="Loan_Types" name="Loan_Types" value="Personal" required><br><br>

      <label for="Number_of_Open_Accounts">Number of Open Accounts:</label><br>
      <input type="number" id="Number_of_Open_Accounts" name="Number_of_Open_Accounts" value="10" required><br><br>

      <label for="Risk_Status">Risk Status:</label><br>
      <input type="text" id="Risk_Status" name="Risk_Status" value="Medium" required><br><br>

      <input type="submit" value="Calculate Credit Score">
    </form>
  </body>
</html>
"""

sector_template = """
<!doctype html>
<html lang="en">
  <head>
    <title>Sector Decision</title>
  </head>
  <body>
    <h1>Predicted Credit Score: {{ predicted_score }}</h1>
    <h2>Select a Sector for Decision</h2>
    <form action="{{ url_for('apply_rules_ui') }}" method="post">
      <label for="sector">Choose a sector:</label><br>
      <select id="sector" name="sector" required>
        <option value="loan">Loan</option>
        <option value="insurance">Insurance</option>
        <option value="real estate">Real Estate</option>
      </select><br><br>
      <!-- Pass the predicted score as a hidden field -->
      <input type="hidden" name="predicted_credit_score" value="{{ predicted_score }}">
      <input type="submit" value="Get Decision">
    </form>
  </body>
</html>
"""

result_template = """
<!doctype html>
<html lang="en">
  <head>
    <title>Decision Result</title>
  </head>
  <body>
    <h1>Sector: {{ sector|capitalize }}</h1>
    <h2>Decision: {{ decision }}</h2>
    {% if details %}
      <p>{{ details }}</p>
    {% endif %}
    <br>
    <a href="{{ url_for('home') }}">Back to Home</a>
  </body>
</html>
"""

# -------------------------------
# UI Routes
# -------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template_string(home_template)

@app.route("/predict", methods=["POST"])
def predict_credit_score_ui():
    try:
        # Retrieve form data and create a new_sample dictionary
        new_sample = {
            "Age": int(request.form["Age"]),
            "Gender": request.form["Gender"],
            "Risk Ratings": request.form["Risk_Ratings"],
            "Location": request.form["Location"],
            "Delinquency Frequency": int(request.form["Delinquency_Frequency"]),
            "Marital Status": request.form["Marital_Status"],
            "Monthly Income (NGN)": float(request.form["Monthly_Income"]),
            "Debt-to-Income Ratio": float(request.form["Debt_to_Income_Ratio"]),
            "Working Sector": request.form["Working_Sector"],
            "Loan Types": request.form["Loan_Types"],
            "Number of Open Accounts": int(request.form["Number_of_Open_Accounts"]),
            "Risk Status": request.form["Risk_Status"]
        }
        sample_df = pd.DataFrame([new_sample])
        # Preprocess the input
        sample_transformed = credit_preprocessor.transform(sample_df)
        # Get prediction
        predicted_score = credit_model.predict(sample_transformed)[0]
    except Exception as e:
        flash(f"Error processing your input: {e}")
        return redirect(url_for('home'))
    
    # Render sector selection form passing the predicted score
    return render_template_string(sector_template, predicted_score=predicted_score)

@app.route("/apply", methods=["POST"])
def apply_rules_ui():
    if "sector" not in request.form or "predicted_credit_score" not in request.form:
        flash("Missing required information. Please try again.")
        return redirect(url_for('home'))
    
    sector = request.form["sector"]
    try:
        predicted_credit_score = float(request.form["predicted_credit_score"])
    except ValueError:
        flash("Invalid credit score value.")
        return redirect(url_for('home'))

    decision = ""
    details = ""

    # Apply rules based on sector
    if sector.lower() == "loan":
        if predicted_credit_score >= 750:
            decision = "Approved"
            details = "Excellent credit score. Eligible for favorable loan terms."
        elif predicted_credit_score >= 650:
            decision = "Conditional Approval"
            details = "Moderate credit score. May require additional guarantees."
        else:
            decision = "Rejected"
            details = "Low credit score. Not eligible for a loan."
    elif sector.lower() == "insurance":
        if predicted_credit_score >= 700:
            decision = "Standard Premium"
        else:
            decision = "Higher Premium"
            details = "Credit score below standard threshold; higher risk premium applies."
    elif sector.lower() == "real estate":
        if predicted_credit_score >= 720:
            decision = "Eligible for Mortgage"
        else:
            decision = "Not Eligible"
            details = "Credit score too low for mortgage eligibility."
    else:
        flash("Invalid sector specified.")
        return redirect(url_for('home'))

    return render_template_string(result_template, sector=sector, decision=decision, details=details)

# -------------------------------
# JSON API Endpoints (if still needed)
# -------------------------------

'''@app.route("/predict-credit-score", methods=["POST"])
def predict_credit_score_api():
    data = request.get_json()
    # Using data from request or a sample (for demonstration)
    new_sample = {
        "Age": 30,
        "Gender": "Male",
        "Risk Ratings": "Medium",
        "Location": "Lagos",
        "Delinquency Frequency": 3,
        "Marital Status": "Single",
        "Monthly Income (NGN)": 500000,
        "Debt-to-Income Ratio": 0.5,
        "Working Sector": "Finance",
        "Loan Types": "Personal",
        "Number of Open Accounts": 10,
        "Risk Status": "Medium"
    }
    sample_df = pd.DataFrame([new_sample])
    try:
        sample_transformed = credit_preprocessor.transform(sample_df)
    except Exception as e:
        return {"error": f"Preprocessing error: {e}"}, 500
    predicted_score = credit_model.predict(data)[0]
    return {"predicted_credit_score": predicted_score}'''


@app.route("/predict-credit-score", methods=["POST"])
def predict_credit_score_api():
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return {"error": "No JSON data received"}, 400
    
    try:
        # Convert the received data to a DataFrame
        sample_df = pd.DataFrame([data])
        
        # Preprocess the data
        sample_transformed = credit_preprocessor.transform(sample_df)
        
        # Make prediction
        predicted_score = credit_model.predict(sample_transformed)[0]
        
        return {"predicted_credit_score": float(predicted_score)}  # Convert to float for JSON serialization
    
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}, 500

        

@app.route("/apply-rules", methods=["POST"])
def apply_rules_api():
    data = request.get_json()
    if not data or "sector" not in data or "predicted_credit_score" not in data:
        return {"error": "Please provide 'sector' and 'predicted_credit_score' in the request body."}, 400
    
    sector = data["sector"]
    predicted_credit_score = data["predicted_credit_score"]
    
    decision = ""
    details = ""
    
    if sector.lower() == "loan":
        if predicted_credit_score >= 750:
            decision = "Approved"
            details = "Excellent credit score. Eligible for favorable loan terms."
        elif predicted_credit_score >= 650:
            decision = "Conditional Approval"
            details = "Moderate credit score. May require additional guarantees."
        else:
            decision = "Rejected"
            details = "Low credit score. Not eligible for a loan."
    elif sector.lower() == "insurance":
        if predicted_credit_score >= 700:
            decision = "Standard Premium"
        else:
            decision = "Higher Premium"
            details = "Credit score below standard threshold; higher risk premium applies."
    elif sector.lower() == "real estate":
        if predicted_credit_score >= 720:
            decision = "Eligible for Mortgage"
        else:
            decision = "Not Eligible"
            details = "Credit score too low for mortgage eligibility."
    else:
        return {"error": "Invalid sector specified. Choose loan, insurance, or real estate."}, 400
    
    return {"sector": sector, "decision": decision, "details": details}

#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
