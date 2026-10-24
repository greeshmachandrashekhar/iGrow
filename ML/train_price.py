import pandas as pd
import joblib
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "datasets/price/price_data.csv"


print("\n" + "=" * 60)
print("iGROW PRICE PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

# We are starting with Tomato only
df = df[df["crop"] == "tomato"].copy()

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

print("Crop selected: Tomato")
print("Rows:", len(df))


# ============================================================
# 2. CREATE DATE FEATURES
# ============================================================

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month


# ============================================================
# 3. CREATE PRICE LAG FEATURES
# ============================================================

# Previous 7 days' prices

df["price_lag_1"] = (
    df["modal_price_per_quintal"].shift(1)
)

df["price_lag_2"] = (
    df["modal_price_per_quintal"].shift(2)
)

df["price_lag_3"] = (
    df["modal_price_per_quintal"].shift(3)
)

df["price_lag_4"] = (
    df["modal_price_per_quintal"].shift(4)
)

df["price_lag_5"] = (
    df["modal_price_per_quintal"].shift(5)
)

df["price_lag_6"] = (
    df["modal_price_per_quintal"].shift(6)
)

df["price_lag_7"] = (
    df["modal_price_per_quintal"].shift(7)
)


# ============================================================
# 4. CREATE TARGET
# ============================================================

# Target = next day's tomato price

df["next_day_price"] = (
    df["modal_price_per_quintal"].shift(-1)
)


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

df = df.dropna().reset_index(drop=True)

print("\nData after preparation:")
print("Rows:", len(df))


# ============================================================
# 6. SELECT FEATURES
# ============================================================

features = [

    # Historical prices
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_4",
    "price_lag_5",
    "price_lag_6",
    "price_lag_7",

    # Weather
    "temperature",
    "humidity",
    "rainfall",

    # Date
    "day",
    "month"
]


X = df[features]

y = df["next_day_price"]


print("\nFeatures used:")
for feature in features:
    print("-", feature)


# ============================================================
# 7. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# We do NOT randomly shuffle time-series data.

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 8. CREATE XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost...")


model = XGBRegressor(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    objective="reg:squarederror"
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 11. EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 4))


# ============================================================
# 12. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

comparison = pd.DataFrame({

    "Actual Price":
        y_test.iloc[:10].values,

    "Predicted Price":
        predictions[:10]

})

comparison["Predicted Price"] = (
    comparison["Predicted Price"].round(2)
)

print(comparison)


# ============================================================
# 13. SAVE MODEL
# ============================================================

os.makedirs(
    "saved_models",
    exist_ok=True
)


joblib.dump(
    model,
    "saved_models/price_model.pkl"
)


# ============================================================
# 14. SAVE FEATURE LIST
# ============================================================

joblib.dump(
    features,
    "saved_models/price_features.pkl"
)


print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("Model:")
print("saved_models/price_model.pkl")

print("\nFeatures:")
print("saved_models/price_features.pkl")


print("\nTraining completed successfully!")