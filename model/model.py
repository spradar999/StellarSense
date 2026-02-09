import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------
# 1. Load dataset
# -------------------------
data = "star_classification.csv"   # change path if needed
df = pd.read_csv(data)

# -------------------------
# 2. Define target
# -------------------------
target = "class"
X = df.drop(columns=[target, "obj_ID", "spec_obj_ID"], errors="ignore")
y = df[target]

# -------------------------
# 3. Encode target
# -------------------------
le = LabelEncoder()
y = le.fit_transform(y)

# -------------------------
# 4. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 5. Scaling
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 6. Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
}

# -------------------------
# 7. Training + Evaluation
# -------------------------
results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    results.append([name, acc, auc, prec, rec, f1, mcc])

# -------------------------
# 8. Results table
# -------------------------
results_df = pd.DataFrame(results, columns=["Model","Accuracy","AUC","Precision","Recall","F1","MCC"])
print(results_df.sort_values(by="Accuracy", ascending=False))

# -------------------------
# 9. Train on FULL DATA & Save Models
# -------------------------
X_full_scaled = scaler.fit_transform(X)

models_save = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
}

for name, model in models_save.items():
    if name in ["logistic_regression", "knn"]:
        model.fit(X_full_scaled, y)
    else:
        model.fit(X, y)
    joblib.dump(model, f"{name}.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ All models saved successfully.")
