# 🚢 RMS Titanic Survival Predictor

A fast, interactive, and modern web application built on top of your trained **Random Forest model** (`model.pkl`) to predict passenger survival on the Titanic.

---

## 🌟 Features

- **Interactive Web Interface**: Clean nautical-themed dashboard with responsive controls for class, sex, age, family members, fare, and embarkation port.
- **Historical Presets**: One-click test scenarios for famous historical passengers:
  - 🌹 **Rose DeWitt Bukater** (1st Class, 17yo Female)
  - 🎨 **Jack Dawson** (3rd Class, 20yo Male)
  - 🎩 **John Jacob Astor IV** (1st Class, 47yo Male)
  - 👨‍👩‍👧‍👦 **3rd Class Family** (Steerage Mother traveling with children)
- **Detailed Probability Analysis**: Displays both **Survival Probability %** and **Mortality Probability %** with animated gauges and survival verdicts.
- **Explainable Factors**: Highlights key contributing factors (e.g. "Women and children first" evacuation priority, deck accessibility, traveling with family).
- **REST API Support**: Built-in JSON endpoint (`/api/predict`) for programmatic integration.
- **Terminal CLI Mode**: Interactive command-line mode for quick terminal predictions (`python app.py --cli`).

---

## 🚀 How to Run

### Method 1: Double-click Launcher (Windows)
Double-click **`run_app.bat`**. It will launch the application and open `http://127.0.0.1:5000` automatically in your web browser.

### Method 2: Command Line (Web UI)
Open a terminal in this folder and run:
```bash
python app.py
```
Then navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Method 3: Command Line (Interactive CLI)
If you prefer testing directly in the terminal without a browser:
```bash
python app.py --cli
```

---

## 🔌 API Usage

### `POST /api/predict`
Send passenger manifest data as JSON to receive a prediction:

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 22,
    "sibsp": 1,
    "parch": 0,
    "fare": 120.0,
    "embarked": "C"
  }'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "survived": true,
    "verdict": "SURVIVED",
    "survival_probability": 86.0,
    "mortality_probability": 14.0,
    "factors": [
      "Female passenger priority (historical 'women and children first' policy).",
      "1st Class ticket: Upper deck proximity and faster lifeboat access.",
      "Small family group (2 members): Mutual assistance during boarding."
    ]
  }
}
```

---

## 🧠 Model Architecture & Feature Engineering

The pre-trained model (`model.pkl`) is a Scikit-Learn **Random Forest Classifier** (`n_estimators=100`, `random_state=42`) trained on the standardized Titanic dataset.

The model expects **9 standardized features**:
1. `Pclass`: Ticket class (1st = 1, 2nd = 2, 3rd = 3)
2. `Age`: Passenger age in years
3. `SibSp`: Number of siblings / spouses aboard
4. `Parch`: Number of parents / children aboard
5. `Fare`: Passenger ticket fare in £
6. `FamilySize`: Total group size (`SibSp + Parch + 1`)
7. `Sex_male`: One-hot encoded gender (1 = Male, 0 = Female)
8. `Embarked_Q`: Embarked at Queenstown (1 = Yes, 0 = No)
9. `Embarked_S`: Embarked at Southampton (1 = Yes, 0 = No)

The app automatically handles input parsing, family size calculation, dummy encoding, and StandardScaler standardization matching the exact training parameters.

---

## 🧪 Running Tests
To verify all predictions, routes, and edge cases:
```bash
python test_app.py
```

