from sklearn.model_selection import train_test_split


class DataSplitter:

    def split(self, df, target_column, test_size, random_state):

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print("\nData Split Successfully")

        print("Training data:", X_train.shape)
        print("Testing data :", X_test.shape)

        return X_train, X_test, y_train, y_test