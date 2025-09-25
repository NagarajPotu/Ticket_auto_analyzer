import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)      # Remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_dataset(path: str):
    df = pd.read_csv(path)
    df["text"] = (df["Subject"].fillna("") + " " + df["Description"].fillna("")).map(clean_text)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text"], df["label"], test_size=0.3, stratify=df["label"], random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), le
