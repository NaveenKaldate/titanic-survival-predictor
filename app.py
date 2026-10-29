"""
Titanic Survival Prediction Web Application
Powered by trained Random Forest Model (model.pkl) and Flask.
"""

import os
import sys
import warnings

# Suppress scikit-learn unpickling warnings
warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

import pickle
import webbrowser
from threading import Timer
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Exact StandardScaler parameters fitted during training
SCALER_MEAN = np.array([
    2.308641975308642,   # 0: Pclass
    29.36158249158249,   # 1: Age
    0.5230078563411896,  # 2: SibSp
    0.38159371492704824, # 3: Parch
    32.204207968574636,  # 4: Fare
    1.904601571268238,   # 5: FamilySize (SibSp + Parch + 1)
    0.6475869809203143,  # 6: Sex_male (1=Male, 0=Female)
    0.08641975308641975, # 7: Embarked_Q
    0.7250280583613917   # 8: Embarked_S
], dtype=np.float64)

SCALER_SCALE = np.array([
    0.8356019334795166,  # 0: Pclass
    13.012388272793666,  # 1: Age
    1.1021244350892878,  # 2: SibSp
    0.8056047612452208,  # 3: Parch
    49.6655344447741,    # 4: Fare
    1.6125528671095162,  # 5: FamilySize
    0.47772176322921145, # 6: Sex_male
    0.2809828809071862,  # 7: Embarked_Q
    0.4465001376820639   # 8: Embarked_S
], dtype=np.float64)

# Load the trained model
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return model

def predict_passenger(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Transforms passenger inputs using exact scaling and predicts survival.
    Returns a dictionary with prediction details, probabilities, and factors.
    """
    rf_model = load_model()

    # Preprocess & normalize inputs
    pclass = int(pclass)
    age = float(age)
    sibsp = int(sibsp)
    parch = int(parch)
    fare = float(fare)
    family_size = sibsp + parch + 1

    sex_str = str(sex).strip().lower()
    sex_male = 1.0 if sex_str in ["male", "m", "1"] else 0.0

    emb_str = str(embarked).strip().upper()
    embarked_q = 1.0 if emb_str == "Q" else 0.0
    embarked_s = 1.0 if emb_str == "S" else 0.0

    raw_features = np.array([
        pclass,
        age,
        sibsp,
        parch,
        fare,
        family_size,
        sex_male,
        embarked_q,
        embarked_s
    ], dtype=np.float64)

    # Standardize using training mean & scale
    scaled_features = (raw_features - SCALER_MEAN) / SCALER_SCALE
    scaled_2d = scaled_features.reshape(1, -1)

    prediction = int(rf_model.predict(scaled_2d)[0])
    probabilities = rf_model.predict_proba(scaled_2d)[0]

    prob_deceased = float(probabilities[0])
    prob_survived = float(probabilities[1])

    # Human-readable insights
    factors = []
    if sex_male == 0.0:
        factors.append("Female passenger priority (historical 'women and children first' policy).")
    else:
        factors.append("Male passenger (lower historical evacuation priority).")

    if pclass == 1:
        factors.append("1st Class ticket: Upper deck proximity and faster lifeboat access.")
    elif pclass == 2:
        factors.append("2nd Class ticket: Moderate lifeboat accessibility.")
    else:
        factors.append("3rd Class ticket: Lower deck quarters with restricted deck access.")

    if age < 12:
        factors.append(f"Child passenger (Age {int(age)}): Priority boarding for lifeboats.")
    elif age > 60:
        factors.append(f"Senior passenger (Age {int(age)}): Mobility challenges during emergency.")

    if family_size > 4:
        factors.append(f"Large family group ({family_size} members): Difficult to stay together during evacuation.")
    elif family_size > 1:
        factors.append(f"Small family group ({family_size} members): Mutual assistance during boarding.")
    else:
        factors.append("Traveling alone (Solo passenger).")

    return {
        "survived": bool(prediction == 1),
        "prediction_code": prediction,
        "verdict": "SURVIVED" if prediction == 1 else "DID NOT SURVIVE",
        "survival_probability": round(prob_survived * 100, 1),
        "mortality_probability": round(prob_deceased * 100, 1),
        "inputs": {
            "pclass": pclass,
            "sex": "Male" if sex_male == 1.0 else "Female",
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "family_size": family_size,
            "embarked": "Cherbourg (C)" if (embarked_q == 0 and embarked_s == 0) else ("Queenstown (Q)" if embarked_q == 1 else "Southampton (S)")
        },
        "factors": factors
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {
        "pclass": "3",
        "sex": "female",
        "age": "24",
        "sibsp": "0",
        "parch": "0",
        "fare": "15.0",
        "embarked": "S"
    }

    if request.method == "POST":
        try:
            form_data = {
                "pclass": request.form.get("pclass", "3"),
                "sex": request.form.get("sex", "male"),
                "age": request.form.get("age", "25"),
                "sibsp": request.form.get("sibsp", "0"),
                "parch": request.form.get("parch", "0"),
                "fare": request.form.get("fare", "10.0"),
                "embarked": request.form.get("embarked", "S")
            }
            result = predict_passenger(
                pclass=form_data["pclass"],
                sex=form_data["sex"],
                age=form_data["age"],
                sibsp=form_data["sibsp"],
                parch=form_data["parch"],
                fare=form_data["fare"],
                embarked=form_data["embarked"]
            )
        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result, form_data=form_data)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or request.form
    try:
        pclass = data.get("pclass", 3)
        sex = data.get("sex", "male")
        age = data.get("age", 25)
        sibsp = data.get("sibsp", 0)
        parch = data.get("parch", 0)
        fare = data.get("fare", 10.0)
        embarked = data.get("embarked", "S")

        res = predict_passenger(pclass, sex, age, sibsp, parch, fare, embarked)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/info", methods=["GET"])
def api_info():
    return jsonify({
        "status": "online",
        "model_type": "RandomForestClassifier",
        "features": [
            "Pclass", "Age", "SibSp", "Parch", "Fare",
            "FamilySize", "Sex_male", "Embarked_Q", "Embarked_S"
        ]
    })

def run_cli():
    """Interactive command-line interface."""
    load_model()
    print("=" * 60)
    print("      TITANIC SURVIVAL PREDICTION - CLI MODE")
    print("=" * 60)
    try:
        pclass = input("Ticket Class [1 = 1st, 2 = 2nd, 3 = 3rd] (default: 3): ").strip() or "3"
        sex = input("Gender [male / female] (default: male): ").strip() or "male"
        age = input("Age in years (default: 25): ").strip() or "25"
        sibsp = input("Siblings/Spouses aboard (default: 0): ").strip() or "0"
        parch = input("Parents/Children aboard (default: 0): ").strip() or "0"
        fare = input("Ticket Fare in £ (e.g. 15.0): ").strip() or "15.0"
        embarked = input("Port of Embarkation [C, Q, S] (default: S): ").strip() or "S"

        res = predict_passenger(pclass, sex, age, sibsp, parch, fare, embarked)
        print("\n" + "-" * 40)
        print(f"PREDICTION RESULT : {res['verdict']}")
        print(f"Survival Chance   : {res['survival_probability']}%")
        print(f"Mortality Chance  : {res['mortality_probability']}%")
        print("-" * 40)
        print("Contributing Factors:")
        for f in res["factors"]:
            print(f"  • {f}")
        print("=" * 60)
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        # Start web application
        print("\n" + "=" * 60)
        print(" Titanic Survival Prediction App")
        print(" Server running on: http://127.0.0.1:5000")
        print(" Press CTRL+C to stop.")
        print("=" * 60 + "\n")
        # Optional: auto-open browser after 1 second
        if "--no-browser" not in sys.argv:
            Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
        app.run(host="0.0.0.0", port=5000, debug=False)
