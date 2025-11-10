📊 Customer Churn Prediction Project

This project predicts whether a customer is likely to leave (churn) based on past behavior.
It uses Python, Scikit-learn, and a Logistic Regression model.

✅ Project Overview

Telecom companies often lose customers without warning.
This project analyzes customer data and builds a machine-learning model to predict churn.
Businesses can use this to take preventive action.

📁 Project Structure
customer-churn-project/
│
├── data/
│   └── raw/
│       └── customer_churn.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md

⚙️ Installation
1) Clone the repository
git clone https://github.com/AliBaig98/customer-churn-project.git
cd customer-churn-project

2) Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows

3) Install dependencies
pip install -r requirements.txt

▶️ How to Run
1) Train the model
python -m src.train


✔ Trains the model
✔ Saves the trained model

2) Make prediction
python -m src.predict


✔ Loads saved model
✔ Generates prediction example

📈 Model Used

Logistic Regression

Achieved approx: ~80% accuracy

🗂 Data

Dataset is stored in:

data/raw/customer_churn.csv

🔍 Key Features Used

Tenure

Monthly Charges

Contract Type

Online Services

Tech Support

Payment Method
…and more

✅ Output Summary

Trained model saved

Predict script shows sample prediction

Helpful for churn-reduction strategies

📦 Requirements

Python 3.9+

pandas

numpy

scikit-learn

(Automatically installed via requirements.txt)

🚀 Future Improvements

Support more models (Random Forest / XGBoost)

Deploy on web (FastAPI / Streamlit)

Improve feature selection

👨‍💻 Author

Ali Baig
