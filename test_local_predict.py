from src.predict import predict_single

sample_input = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "Credit": "no",
    "balance": 1500,
    "housing_loan": "yes",
    "personal_loan": "no",
    "contact": "cellular",
    "last_contact_day": 12,
    "last_contact_month": "may",
    "last_contact_duration/sec": "180 sec",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "previous marketing campaign": "unknown"
}

result = predict_single(sample_input)
print(result)