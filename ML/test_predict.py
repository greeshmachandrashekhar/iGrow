import joblib

encoder = joblib.load("saved_models/label_encoder.pkl")

print(encoder.classes_)