import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocessing import prepare_dataset
import numpy as np

MODEL_NAME = "distilbert-base-uncased"

(X_train, y_train), (X_val, y_val), (X_test, y_test), le = prepare_dataset("data/raw/tickets.csv")

train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
val_ds = Dataset.from_dict({"text": X_val.tolist(), "label": y_val.tolist()})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="weighted", zero_division=0)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

training_args = TrainingArguments(
    output_dir="models/transformer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/transformer")
tokenizer.save_pretrained("models/transformer")
print("✅ Transformer model saved!")
