import pandas as pd


class ModelComparison:

    def save_results(self, results, filename):

        df = pd.DataFrame(
            list(results.items()),
            columns=["Model", "Accuracy"]
        )

        df = df.sort_values(
            by="Accuracy",
            ascending=False
        )

        df.to_csv(filename, index=False)

        print(f"\nResults saved to {filename}")

        return df