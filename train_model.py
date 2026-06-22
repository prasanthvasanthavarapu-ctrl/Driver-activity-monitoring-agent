from sklearn.ensemble import RandomForestClassifier

import joblib

joblib.dump(model, "mental_health_model.pkl")