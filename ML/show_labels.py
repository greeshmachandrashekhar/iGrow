from preprocessing.loader import DataLoader
from preprocessing.cleaner import DataCleaner
from sklearn.preprocessing import LabelEncoder

loader = DataLoader("datasets/crop/Crop_recommendation.csv")

df = loader.load()
df = DataCleaner().clean(df)

encoder = LabelEncoder()
encoder.fit(df["label"])

print("\nCrop Label Mapping:\n")

for i, crop in enumerate(encoder.classes_):
    print(f"{i} -> {crop}")