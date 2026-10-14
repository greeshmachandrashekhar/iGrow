import pandas as pd
import os


class DataLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        if self.file_path.endswith(".xlsx"):
            df = pd.read_excel(self.file_path, sheet_name="Sheet1")
        elif self.file_path.endswith(".csv"):
            df = pd.read_csv(self.file_path)
        else:
            raise ValueError("Unsupported file format")

        print("\nDataset Loaded Successfully!")
        print("Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())

        return df