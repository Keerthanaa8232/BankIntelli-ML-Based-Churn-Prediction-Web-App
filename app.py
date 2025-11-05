from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Model Files
model = pickle.load(open("model/xgb_model.pkl","rb"))
label_encoders = pickle.load(open("model/label_encoders.pkl","rb"))
columns = pickle.load(open("model/columns.pkl","rb"))

# HOME PAGE
@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")

# ABOUT PAGE
@app.route('/about')
def about():
    return render_template("about.html")

# PREDICTION PAGE UI
@app.route('/prediction')
def prediction_page():
    return render_template("index.html")

# PREDICTION LOGIC (NO CHANGE)
@app.route('/predict', methods=['POST'])
def predict():
    data = {key: request.form[key] for key in request.form}
    df = pd.DataFrame([data])
    df = df.reindex(columns=columns, fill_value=0)

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "✅ Customer is NOT likely to CHURN"
    else:
        result = "⚠️ Customer is likely to CHURN"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
