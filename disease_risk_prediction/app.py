from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = request.form.get("gender", "")

    pregnancies = request.form.get("pregnancies", "")
    if gender != "female":
        pregnancies = 0

    data = [
        safe_float(pregnancies),
        safe_float(request.form.get("glucose")),
        safe_float(request.form.get("bloodpressure")),
        safe_float(request.form.get("skinthickness")),
        safe_float(request.form.get("insulin")),
        safe_float(request.form.get("bmi")),
        safe_float(request.form.get("dpf")),
        safe_float(request.form.get("age"))
    ]

    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    # 🔥 probability of diabetes
    prob = model.predict_proba(data)[0][1]

    # ============================
    # ✅ FIXED RISK LOGIC (REALISTIC)
    # ============================

    if prob < 0.25:
        risk = "LOW"
        message = "🟢 Low Risk of Diabetes"

    elif prob < 0.55:
        risk = "MEDIUM"
        message = "🟠 Medium Risk of Diabetes"

    else:
        risk = "HIGH"
        message = "🔴 High Risk of Diabetes"

    return render_template(
        "result.html",
        risk=risk,
        message=message,
        prob=round(prob * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)