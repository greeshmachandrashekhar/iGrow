from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ModelFactory:

    @staticmethod
    def get_models():

        models = {

            "Decision Tree":
                DecisionTreeClassifier(random_state=42),

            "Random Forest":
                RandomForestClassifier(random_state=42),

            "Extra Trees":
                ExtraTreesClassifier(random_state=42),

            "Gradient Boosting":
                GradientBoostingClassifier(random_state=42),

            "KNN":
                KNeighborsClassifier(),

            "SVM":
                SVC(),

            "Naive Bayes":
                GaussianNB(),

            "XGBoost":
                XGBClassifier(
                    eval_metric="mlogloss",
                    random_state=42
                ),

            "LightGBM":
                LGBMClassifier(random_state=42),

            "CatBoost":
                CatBoostClassifier(
                    verbose=False,
                    random_state=42
                )
        }

        return models