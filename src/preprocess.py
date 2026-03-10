import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

RAW_TO_INTERNAL_COLUMNS = {
    "Credit": "credit",
    "previous marketing campaign": "previous_marketing_campaign",
    "previous_marketing_campaign": "previous_marketing_campaign",
    "subscribed term deposit": "subscribed_term_deposit",
    "last_contact_duration/sec": "last_contact_duration_sec",
    "last_contact_duration_sec": "last_contact_duration_sec",
}

EDUCATION_MAP = {
    "unknown": 0,
    "primary": 1,
    "secondary": 2,
    "tertiary": 3
}

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

YES_NO_MAP = {
    "no": 0,
    "yes": 1,
}

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns = RAW_TO_INTERNAL_COLUMNS, inplace = True)
    return df

def lowercase_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if is_object_dtype(df[col]) or is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df

def fix_duration_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "last_contact_duration_sec" in df.columns:
        df["last_contact_duration_sec"] = (
            df["last_contact_duration_sec"]
            .astype(str)
            .str.split()
            .str[0]
            .astype(int)
        )
    return df

def encode_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    yes_no_cols = ["credit", "housing_loan", "personal_loan"]

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map(YES_NO_MAP)

    if "subscribed_term_deposit" in df.columns:
        df["subscribed_term_deposit"] = df["subscribed_term_deposit"].map(YES_NO_MAP)

    return df

def map_orderd_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "education" in df.columns:
        df["education"] = df["education"].map(EDUCATION_MAP)

    if "last_contact_month" in df.columns:
        df["last_contact_month"] = df["last_contact_month"].map(MONTH_MAP)

    return df

def one_hot_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    one_hot_cols = ["job", "marital", "contact", "previous_marketing_campaign"]
    available_cols = [col for col in one_hot_cols if col in df.columns]

    df = pd.get_dummies(df, columns = available_cols, drop_first = True)

    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = lowercase_object_columns(df)
    df = fix_duration_column(df)
    df = encode_yes_no_columns(df)
    df = map_orderd_columns(df)
    df = one_hot_encode_columns(df)
    return df

def align_features(df: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df = df.reindex(columns = training_columns, fill_value = 0)
    return df