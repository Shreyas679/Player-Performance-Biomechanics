# ml/train.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

def load_features(root="."):
    dfX = pd.read_csv(Path(root)/"data/features/features.csv", index_col=0)
    y = pd.read_csv(Path(root)/"data/labels.csv")
    y = y.set_index("clip_id").loc[dfX.index]
    return dfX, y

def train_all(root="."):
    X, y = load_features(root)
    features = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # Targets
    y_perf = y["performance_score"].values
    y_inj = y["injury_risk"].values

    Xtr, Xva, yptr, ypva = train_test_split(Xs, y_perf, test_size=0.2, random_state=42)
    Xtrc, Xvac, yitr, yiva = train_test_split(Xs, y_inj, test_size=0.2, random_state=42)

    reg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)
    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss")

    reg.fit(Xtr, yptr)
    clf.fit(Xtrc, yitr)

    pr = reg.predict(Xva)
    pc = clf.predict(Xvac)

    print("Performance R2:", r2_score(ypva, pr))
    print("Performance MAE:", mean_absolute_error(ypva, pr))
    print("Injury Acc:", accuracy_score(yiva, pc))
    print("Injury F1:", f1_score(yiva, pc))

    out = Path(root)/"models"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out/"scaler.pkl")
    reg.save_model(str(out/"xgb_performance.json"))
    clf.save_model(str(out/"xgb_injury.json"))
    with open(out/"feature_list.json","w") as f:
        json.dump(features, f)
    print("Models saved.")

if __name__ == "__main__":
    train_all(".")
