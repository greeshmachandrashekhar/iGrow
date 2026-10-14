from sklearn.preprocessing import StandardScaler
import joblib
import os


class DataScaler:

    def __init__(self):
        self.scaler = StandardScaler()

    def scale(self, X_train):

        X_train_scaled = self.scaler.fit_transform(X_train)

        os.makedirs("saved_models", exist_ok=True)

        joblib.dump(
            self.scaler,
            "saved_models/scaler.pkl"
        )

        print("\nScaler fitted and saved.")

        return X_train_scaled