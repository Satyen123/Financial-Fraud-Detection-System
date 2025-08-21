import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# -------------------------------------
# Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="💳 Financial Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------
# Custom CSS for Better Styling 🎨
# -------------------------------------
st.markdown(
    """
    <style>
    /* Main title style */
    .main-title {
        color: #2E86C1;
        text-align: center;
        font-size: 36px !important;
        font-weight: bold;
    }

    /* Subtitle style */
    .subtitle {
        color: #28B463;
        text-align: center;
        font-size: 18px;
        margin-bottom: 15px;
    }

    /* Success, error, and info box enhancements */
    .stSuccess {background-color: #D4EFDF !important;}
    .stError {background-color: #FADBD8 !important;}
    .stInfo {background-color: #D6EAF8 !important;}

    /* Dataframe styling */
    .dataframe {
        border: 2px solid #2E86C1;
        border-radius: 10px;
    }

    /* Download button styling */
    .stDownloadButton button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 20px;
        transition: 0.3s;
    }
    .stDownloadButton button:hover {
        background-color: #1B4F72;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------
# Title & Description
# -------------------------------------
st.markdown("<h1 class='main-title'>💳 Financial Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a CSV file to detect fraudulent transactions using your trained model</p>", unsafe_allow_html=True)

# -------------------------------------
# Load Model
# -------------------------------------
model_path = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
try:
    model = joblib.load(model_path)
except:
    st.error("❌ Model file not found! Please make sure `rf_model.pkl` is inside the app folder.")
    st.stop()

# -------------------------------------
# CSV Upload Section
# -------------------------------------
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# -------------------------------------
# Process File if Uploaded
# -------------------------------------
if uploaded_file:
    try:
        # Read uploaded CSV
        data = pd.read_csv(uploaded_file)

        # Create 'scaled_amount' if missing
        if "scaled_amount" not in data.columns:
            if "Amount" in data.columns:
                scaler = StandardScaler()
                data["scaled_amount"] = scaler.fit_transform(data[["Amount"]])
                st.info("ℹ️ 'scaled_amount' column generated automatically.")
            else:
                st.error("❌ CSV must have an 'Amount' column.")
                st.stop()

        # Drop target column if present
        for col in ["Class", "Target", "Fraud", "label"]:
            if col in data.columns:
                data = data.drop(col, axis=1)
                st.info(f"ℹ️ Dropped target column: '{col}'")

        # Drop 'Amount' if 'scaled_amount' exists
        if "Amount" in data.columns and "scaled_amount" in data.columns:
            data = data.drop("Amount", axis=1)
            st.info("ℹ️ Dropped 'Amount' column since model uses 'scaled_amount'.")

        # Check feature mismatch
        if data.shape[1] != model.n_features_in_:
            st.error(
                f"❌ Model expects {model.n_features_in_} features, "
                f"but uploaded CSV has {data.shape[1]} features after adjustments."
            )
            st.stop()

        # Convert input to NumPy array
        input_array = data.to_numpy()

        # Make predictions
        predictions = model.predict(input_array)
        probabilities = model.predict_proba(input_array)[:, 1]

        # Add results to dataframe
        data["Fraud Prediction"] = predictions
        data["Fraud Probability"] = probabilities

        # Show success message
        st.success("✅ Fraud detection completed successfully!")

        # Expandable Data Preview Section
        with st.expander("🔍 View Prediction Results"):
            st.dataframe(data.head(50))

        # Download Button
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Fraud Detection Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

        # Show stats
        st.subheader("📊 Prediction Summary")
        fraud_count = data["Fraud Prediction"].sum()
        total = len(data)
        st.metric("🔴 Fraudulent Transactions", fraud_count)
        st.metric("🟢 Safe Transactions", total - fraud_count)
        st.metric("📄 Total Records", total)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.warning("⚠️ Please upload a CSV file to start fraud detection.")
