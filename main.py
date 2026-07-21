import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_excel("fav dataset.xlsx",engine="openpyxl")

print("Dataset Shape:", df.shape)

# ==========================================
# REQUIRED COLUMNS
# ==========================================

required_columns = [
    "headline",
    "body text",
    "label"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

# ==========================================
# CREATE COMBINED TEXT
# ==========================================

df["combined_text"] = (
    df["headline"].astype(str)
    + " "
    + df["body text"].astype(str)
)

# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df["combined_text"])

y = df["label"]

# ==========================================
# TRAIN MODEL
# ==========================================

print("Training Label Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# ==========================================
# SAVE FILES
# ==========================================

joblib.dump(model, "label_model.pkl")

joblib.dump(
    vectorizer,
    "truthscope_vectorizer.pkl"
)

# ==========================================
# COMPLETE
# ==========================================

print("\n=================================")
print("TruthScope Training Completed")
print("=================================")

print("\nFiles Created:")
print("label_model.pkl")
print("truthscope_vectorizer.pkl")
print(df.columns.tolist())