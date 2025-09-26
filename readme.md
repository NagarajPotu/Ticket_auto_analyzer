# 🎫 Ticket Auto Analyzer

A Python-based **AI ticket triage system** that automatically classifies and routes customer support tickets with **high accuracy**.  
Built using **DistilBERT**, **FastAPI**, and custom business logic to ensure speed, precision, and scalability.  

---

## ✨ Features
- 🚀 **Processes multiple tickets at once** (batch prediction supported)  
- 🎯 **High accuracy (96%)** with balanced Precision/Recall  
- ⚡ **Low latency (~0.6ms per ticket)**  
- 📊 **Model evaluation metrics available**  
- 🌐 Easy-to-use **FastAPI interface** with `/docs` and `/redoc`  
- 🖼️ Supports screenshots and visualization  

---

## 📂 Project Structure


---

## 🛠️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/NagarajPotu/Ticket_auto_analyzer.git
cd Ticket_auto_analyzer

## 2️⃣Create and activate a virtual environment

python -m venv menv
menv\Scripts\activate   # On Windows
source menv/bin/activate # On Mac/Linux

## 3️⃣ Install dependencies

pip install -r requirements.txt

## 4️⃣ Run the FastAPI server

uvicorn app.main:app --reload

➡️ Visit http://127.0.0.1:8000

➡️ API docs: http://127.0.0.1:8000/docs

➡️ Redoc UI: http://127.0.0.1:8000/docs#/default/predict_batch_predict_batch_post

📌 Example API Requests

✅ Single Ticket Prediction

{
  "ticket_id": "2001",
  "subject": "App crash on login",
  "description": "Whenever I try to log into the app, it crashes immediately after entering my credentials.",
  "pred_label": "Bug Report",
  "score": 0.92,
  "priority": "High",
  "vip": false
}


✅ Batch Prediction

[
  {
    "ticket_id": "2001",
    "subject": "App crash on login",
    "description": "App crashes on login attempt.",
    "pred_label": "Bug Report",
    "score": 0.92,
    "priority": "High",
    "vip": false
  },
  {
    "ticket_id": "2002",
    "subject": "Request for dark mode",
    "description": "Please add a dark theme to the app.",
    "pred_label": "Feature Request",
    "score": 0.55,
    "priority": "Normal",
    "vip": false
  }
]

## 📊 Model Evaluation

On a test set of 50 tickets:

✅ Accuracy: 96.00%

✅ Precision: 96.36%

✅ Recall: 96.00%

✅ F1-Score: 95.87%

⏱️ Average latency: 0.63 ms per ticket

## 🖼️ Screenshots


https://1drv.ms/i/c/6c59a49af058d375/EUAzbmFdy1JFhK2ILTFiENQBEdh9FXAbWV3OJt6CF1P2iw?e=cMubn4
https://1drv.ms/i/c/6c59a49af058d375/EUMwYyjxQaNImegl7l6MNBsBkA6rJyQ1k2-2r9H_ZYMUPA?e=Irgfbz

## 👨‍💻 Author

Nagaraj Potu
📌 AI/ML Enthusiast | Python Developer
nagarajpotu@gmail.com

