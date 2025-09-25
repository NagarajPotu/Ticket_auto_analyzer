# evaluation.py
import pandas as pd
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.router import route_ticket

# Paths (adjust if needed)
MODEL_PATH = "models/baseline/model.joblib"
LABEL_ENCODER_PATH = "models/baseline/label_encoder.joblib"
TEST_CSV_PATH = "data/raw/tickets.csv"  # Make sure this CSV has columns: ticket_id, subject, description, category, priority, vip

# Load model and label encoder
model = joblib.load(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Load test data
df = pd.read_csv(TEST_CSV_PATH)

y_true = df['Category'].values
y_pred = []
latencies = []

for idx, row in df.iterrows():
    text = f"{row['Subject']} {row['Description']}"
    start = time.time()
    proba = model.predict_proba([text])[0]
    score_idx = proba.argmax()
    pred_label = le.classes_[score_idx]
    latency = (time.time() - start) * 1000  # in ms
    latencies.append(latency)
    y_pred.append(pred_label)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
avg_latency = sum(latencies) / len(latencies)

# Print results
print("==== Model Evaluation ====")
print(f"Total tickets evaluated: {len(df)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Average latency (ms): {avg_latency:.2f}")
