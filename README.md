Financial Fraud Detection

A machine learning-based Streamlit web application that detects fraudulent credit card transactions using a trained Random Forest model.
The app allows you to upload a CSV file, processes the data, and predicts whether each transaction is fraudulent or legitimate.

🚀 Features

✅ Upload CSV files with transaction data
✅ Automatic data preprocessing
✅ Top 30 important features used for prediction
✅ Real-time fraud detection
✅ Fraud probability scores for each transaction
✅ Downloadable prediction results
✅ Streamlit-powered interactive web app

🛠️ Tech Stack

Frontend & App: Streamlit

Machine Learning Model: RandomForestClassifier
Backend: Python 3.12
Data Handling: Pandas, NumPy
Model Saving: Joblib
Visualization: Matplotlib, Seaborn

📂 Project Structure
financial-fraud-detection/
│
├── app/
│   ├── app.py                # Streamlit main app file
│   ├── rf_model.pkl          # Trained Random Forest model
│   ├── top_15_features.json  # Top 15 features list
│   ├── requirements.txt      # Required dependencies
│
├── dataset/
│   ├── creditcard.csv        # Training dataset (optional)
│
├── README.md                 # Project documentation
└── .gitignore                # To ignore unnecessary files

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Satyen123/financial-fraud-detection.git
cd financial-fraud-detection/app

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App Locally
streamlit run app.py


The app will start at:
🔗 http://localhost:8501

📌 How to Use

Prepare your CSV file with the same 30 features used to train the model.

Click Upload CSV in the app.

The app will process the data automatically.

View predictions in the browser.

Click Download Result CSV to save the output.

📊 Model Details

Algorithm Used: Random Forest Classifier

Trained On: Financial credit card transaction dataset

Top Features Used:
V1, V2, V3, ..., V28, Amount, scaled_amount

Performance Metric:

ROC-AUC: 0.90 ✅

Accuracy: ~94%
