import joblib
import pandas as pd


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = joblib.load(
    "saved_models/price_model.pkl"
)

features = joblib.load(
    "saved_models/price_features.pkl"
)


print("\n" + "=" * 60)
print("iGROW TOMATO PRICE PREDICTION")
print("=" * 60)


# ============================================================
# 2. GET PREVIOUS 7 DAYS' PRICES
# ============================================================

print("\nEnter the previous 7 days' tomato prices")
print("(₹ per quintal)")

price_lag_1 = float(
    input("1 day ago: ")
)

price_lag_2 = float(
    input("2 days ago: ")
)

price_lag_3 = float(
    input("3 days ago: ")
)

price_lag_4 = float(
    input("4 days ago: ")
)

price_lag_5 = float(
    input("5 days ago: ")
)

price_lag_6 = float(
    input("6 days ago: ")
)

price_lag_7 = float(
    input("7 days ago: ")
)


# ============================================================
# 3. GET WEATHER DATA
# ============================================================

print("\nEnter current weather information")

temperature = float(
    input("Temperature (°C): ")
)

humidity = float(
    input("Humidity (%): ")
)

rainfall = float(
    input("Rainfall (mm): ")
)


# ============================================================
# 4. GET DATE
# ============================================================

print("\nEnter prediction date")

date = input(
    "Date (YYYY-MM-DD): "
)

date = pd.to_datetime(date)

day = date.day
month = date.month


# ============================================================
# 5. CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame([{

    "price_lag_1": price_lag_1,

    "price_lag_2": price_lag_2,

    "price_lag_3": price_lag_3,

    "price_lag_4": price_lag_4,

    "price_lag_5": price_lag_5,

    "price_lag_6": price_lag_6,

    "price_lag_7": price_lag_7,

    "temperature": temperature,

    "humidity": humidity,

    "rainfall": rainfall,

    "day": day,

    "month": month
}])


# Make sure feature order is exactly
# the same as during training

input_data = input_data[features]


# ============================================================
# 6. PREDICT
# ============================================================

prediction = model.predict(
    input_data
)


predicted_price = prediction[0]


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("iGROW PRICE PREDICTION RESULT")
print("=" * 60)

print("Crop                : Tomato")

print(
    "Predicted Price     : ₹",
    round(predicted_price, 2),
    "per quintal"
)

print("=" * 60)
