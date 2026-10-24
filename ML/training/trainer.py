import joblib
from sklearn.metrics import accuracy_score


class ModelTrainer:

    def train(self, models, X_train, X_test, y_train, y_test):

        results = {}

        best_model = None
        best_model_name = None
        best_accuracy = 0

        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)

        for name, model in models.items():

            print(f"Training {name}...")

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            accuracy = accuracy_score(y_test, prediction)

            results[name] = accuracy

            print(f"{name} : {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name

        return results, best_model, best_model_name, best_accuracy