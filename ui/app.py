import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Customer Subscription Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("Customer Subscription Predictor")
st.write("Enter customer details to predict whether the customer will subscribe to the term deposit.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"
        ]
    )

    marital = st.selectbox("Marital Status", ["single", "married", "divorced"])

    education = st.selectbox("Education", ["unknown", "primary", "secondary", "tertiary"])

    credit = st.selectbox("Credit in Default?", ["yes", "no"])

    balance = st.number_input("Balance", value=1500.0)

    housing_loan = st.selectbox("Housing Loan?", ["yes", "no"])

    personal_loan = st.selectbox("Personal Loan?", ["yes", "no"])

    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])

    last_contact_day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=12)

    last_contact_month = st.selectbox(
        "Last Contact Month",
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
    )

    last_contact_duration_sec = st.text_input("Last Contact Duration", value="180 sec")

    campaign = st.number_input("Campaign Contacts", min_value=1, value=2)

    pdays = st.number_input("Pdays", value=-1)

    previous = st.number_input("Previous Contacts", min_value=0, value=0)

    previous_marketing_campaign = st.selectbox(
        "Previous Marketing Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "Credit": credit,
        "balance": balance,
        "housing_loan": housing_loan,
        "personal_loan": personal_loan,
        "contact": contact,
        "last_contact_day": last_contact_day,
        "last_contact_month": last_contact_month,
        "last_contact_duration_sec": last_contact_duration_sec,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "previous_marketing_campaign": previous_marketing_campaign
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.success(f"Prediction: {result['label']}")
            st.write(f"Probability of subscription: {result['probability']}")
            st.write(f"Threshold used: {result['threshold_used']}")

            if result["prediction"] == 1:
                st.info("This customer is predicted to subscribe.")
            else:
                st.warning("This customer is predicted not to subscribe.")
        else:
            st.error("Prediction request failed.")
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Make sure the API server is running on http://127.0.0.1:8000")
    except requests.exceptions.Timeout:
        st.error("The request timed out.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")