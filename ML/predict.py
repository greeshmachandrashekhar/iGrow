import joblib
import pandas as pd


# ---------------------------------
# Load saved files
# ---------------------------------
model = joblib.load("saved_models/crop_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
encoders = joblib.load("saved_models/encoders.pkl")
crop_encoder = joblib.load("saved_models/crop_label_encoder.pkl")


# ---------------------------------
# Get user input
# ---------------------------------
temperature = float(input("Enter temperature: "))
humidity = float(input("Enter humidity: "))
ph = float(input("Enter pH: "))
ec = float(input("Enter EC: "))
light_hours = float(input("Enter light hours: "))

water = input("Enter water availability (high/medium/low): ").lower()
season = input("Enter season (cool/warm): ").lower()


# ---------------------------------
# Encode categorical inputs
# ---------------------------------
water_encoded = encoders["water availablity"].transform([water])[0]
season_encoded = encoders["season"].transform([season])[0]


# ---------------------------------
# Create input dataframe
# ---------------------------------
input_data = pd.DataFrame([[
    temperature,
    humidity,
    ph,
    ec,
    light_hours,
    water_encoded,
    season_encoded
]], columns=[
    "temperature",
    "humidity",
    "ph",
    "ec",
    "light hours",
    "water availablity",
    "season"
])


# ---------------------------------
# Scale input
# ---------------------------------
input_scaled = scaler.transform(input_data)


# ---------------------------------
# Predict crop
# ---------------------------------
prediction = model.predict(input_scaled)


# ---------------------------------
# Convert number → crop name
# ---------------------------------
crop_name = crop_encoder.inverse_transform(prediction)[0]


print("\n" + "=" * 40)
print("iGROW CROP RECOMMENDATION")
print("=" * 40)

print("Recommended Crop:", crop_name)