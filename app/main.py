from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib, time
from src.router import route_ticket

app = FastAPI(title="Ticket Auto-Triage")

# Load baseline model
model = joblib.load("models/baseline/model.joblib")
le = joblib.load("models/baseline/label_encoder.joblib")

class Ticket(BaseModel):
    ticket_id: int
    subject: str
    description: str
    priority: str = "Normal"
    vip: bool = False

@app.get("/")
def read_root():
    return {"message": "Ticket Auto-Triage API is running!"}

@app.post("/predict")
def predict(ticket: Ticket):
    text = f"{ticket.subject} {ticket.description}"
    start = time.time()
    proba = model.predict_proba([text])[0]
    idx = proba.argmax()
    label = le.classes_[idx]
    score = proba[idx]
    route = route_ticket(label, score, ticket.priority, ticket.vip)
    latency = round((time.time() - start) * 1000, 2)

    return {
        "ticket_id": ticket.ticket_id,
        "predicted_category": label,
        "score": float(score),
        "route": route,
        "latency_ms": latency
    }

@app.post("/predict_batch")
def predict_batch(tickets: List[Ticket]):
    results = []
    for ticket in tickets:
        text = f"{ticket.subject} {ticket.description}"
        start = time.time()
        proba = model.predict_proba([text])[0]
        idx = proba.argmax()
        label = le.classes_[idx]
        score = proba[idx]
        route = route_ticket(label, score, ticket.priority, ticket.vip)
        latency = round((time.time() - start) * 1000, 2)
        
        results.append({
            "ticket_id": ticket.ticket_id,
            "predicted_category": label,
            "score": float(score),
            "route": route,
            "latency_ms": latency
        })
    return results
