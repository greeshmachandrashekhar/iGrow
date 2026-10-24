from sklearn.preprocessing import LabelEncoder
import joblib
import os


class DataEncoder:

    def __init__(self):
        self.encoders = {}

    def encode(self, df, target_column):

        df = df.copy()

        # -----------------------------
        # Encode input categorical columns
        # -----------------------------
        for column in df.columns:

            if column == target_column:
                continue

            if df[column].dtype == "object" or df[column].dtype == "str":

                encoder = LabelEncoder()

                df[column] = encoder.fit_transform(df[column])

                self.encoders[column] = encoder

                print(f"\nEncoded: {column}")
                print(dict(zip(
                    encoder.classes_,
                    encoder.transform(encoder.classes_)
                )))

        # -----------------------------
        # Encode crop target
        # -----------------------------
        target_encoder = LabelEncoder()

        df[target_column] = target_encoder.fit_transform(
            df[target_column].astype(str)
        )

        self.encoders[target_column] = target_encoder

        print(f"\nEncoded: {target_column}")
        print(dict(zip(
            target_encoder.classes_,
            target_encoder.transform(target_encoder.classes_)
        )))

        # -----------------------------
        # Save encoders
        # -----------------------------
        os.makedirs("saved_models", exist_ok=True)

        joblib.dump(
            self.encoders,
            "saved_models/encoders.pkl"
        )

        joblib.dump(
            target_encoder,
            "saved_models/crop_label_encoder.pkl"
        )

        print("\nEncoders saved successfully.")

        return df