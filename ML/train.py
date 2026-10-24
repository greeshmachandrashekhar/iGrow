from configs import crop_config

from preprocessing.loader import DataLoader
from preprocessing.cleaner import DataCleaner
from preprocessing.encoder import DataEncoder
from preprocessing.scalar import DataScaler
from preprocessing.splitter import DataSplitter

from models.model_factory import ModelFactory
from training.trainer import ModelTrainer
from evaluation.comparison import ModelComparison

import os
import joblib


def run_pipeline():

    print("\n" + "=" * 60)
    print("iGROW CROP RECOMMENDATION")
    print("=" * 60)

    # ---------------------------------
    # 1. Load dataset
    # ---------------------------------
    loader = DataLoader(crop_config.DATASET_PATH)
    df = loader.load()

    # ---------------------------------
    # 2. Clean dataset
    # ---------------------------------
    cleaner = DataCleaner()
    df = cleaner.clean(df)

    # ---------------------------------
    # 3. Encode categorical data
    # ---------------------------------
    encoder = DataEncoder()
    df = encoder.encode(df, crop_config.TARGET_COLUMN)

    # ---------------------------------
    # 4. Split data
    # ---------------------------------
    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(
        df,
        crop_config.TARGET_COLUMN,
        crop_config.TEST_SIZE,
        crop_config.RANDOM_STATE
    )

    # ---------------------------------
    # 5. Scale features
    # ---------------------------------
    scaler = DataScaler()

    X_train = scaler.scale(X_train)
    X_test = scaler.scaler.transform(X_test)

    # ---------------------------------
    # 6. Create models
    # ---------------------------------
    factory = ModelFactory()
    models = factory.get_models()

    # ---------------------------------
    # 7. Train and compare models
    # ---------------------------------
    trainer = ModelTrainer()

    results, best_model, best_model_name, best_accuracy = trainer.train(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------
    # 8. Show sample predictions
    # ---------------------------------
    print("\nSample Predictions")

    predictions = best_model.predict(X_test[:10])

    print("Predictions :", predictions)
    print("Actual      :", y_test.iloc[:10].values)

    # ---------------------------------
    # 9. Save best model
    # ---------------------------------
    os.makedirs("saved_models", exist_ok=True)

    model_path = "saved_models/crop_model.pkl"

    joblib.dump(
        best_model,
        model_path
    )

    print("\nBest model saved to:", model_path)

    # ---------------------------------
    # 10. Print final result
    # ---------------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print("Best Model :", best_model_name)
    print("Accuracy   :", round(best_accuracy, 4))

    # ---------------------------------
    # 11. Save model comparison
    # ---------------------------------
    os.makedirs("results", exist_ok=True)

    comparison = ModelComparison()

    comparison.save_results(
        results,
        "results/crop_model_results.csv"
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    run_pipeline()