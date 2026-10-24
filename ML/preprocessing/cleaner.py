import pandas as pd


class DataCleaner:

    def clean(self, df):

        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)

        print("\nChecking Missing Values...")
        print(df.isnull().sum())

        print("\nChecking Duplicate Rows...")
        print(df.duplicated().sum())

        print("\nRemoving Duplicate Rows...")
        df = df.drop_duplicates()

        print(f"Remaining Rows : {len(df)}")

        print("\nData Types")
        print(df.dtypes)

        return df