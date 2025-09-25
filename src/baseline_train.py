import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.preprocessing import prepare_dataset

if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test), le = prepare_dataset("data/raw/tickets.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    print("Validation Report:\n", classification_report(y_val, preds, target_names=le.classes_))

    joblib.dump(model, "models/baseline/model.joblib")
    joblib.dump(le, "models/baseline/label_encoder.joblib")
    print("✅ Baseline model saved!")
