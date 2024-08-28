from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

CLEVELAND = 'src/hd/data/dataset/cleveland.data'
HUNGARIAN = 'src/hd/data/dataset/hungarian.data'
SWITZERLAND = 'src/hd/data/dataset/switzerland.data'
VA = 'src/hd/data/dataset/va.data'

class DataSet:
    def __init__(self, cleveland_path=CLEVELAND, hungarian_path=HUNGARIAN, switzerland_path=SWITZERLAND, va_path=VA, test_size=0.2, balance_data=False):
        self.column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
        self.dataset = self.load_and_combine_datasets(cleveland_path, hungarian_path, switzerland_path, va_path)
        self.test_size = test_size
        self.balance_data = balance_data
        self.preprocessor = self.create_preprocessor()
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()

    def load_and_combine_datasets(self, cleveland_path, hungarian_path, switzerland_path, va_path):
        cleveland_df = pd.read_csv(cleveland_path, names=self.column_names, na_values='?')
        hungarian_df = pd.read_csv(hungarian_path, names=self.column_names, na_values='?')
        switzerland_df = pd.read_csv(switzerland_path, names=self.column_names, na_values='?')
        va_df = pd.read_csv(va_path, names=self.column_names, na_values='?')
        return pd.concat([cleveland_df, hungarian_df, switzerland_df, va_df], ignore_index=True)

    def create_preprocessor(self):
        numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def prepare_data(self):
        X = self.dataset.drop("num", axis=1)
        y = (self.dataset["num"] > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        if self.balance_data:
            smote = SMOTE(random_state=42)
            X_train_transformed, y_train = smote.fit_resample(X_train_transformed, y_train)

        return X_train_transformed, X_test_transformed, y_train, y_test

    def calculate_target_percentages(self):
        unique, counts = np.unique(self.y_train, return_counts=True)
        total = len(self.y_train)
        return {
            "No Heart Disease (0)": (counts[0] / total) * 100,
            "Heart Disease (1)": (counts[1] / total) * 100
        }

    def print_target_percentages(self):
        percentages = self.calculate_target_percentages()
        print("Percentuali della variabile target:")
        for label, percentage in percentages.items():
            print(f"{label}: {percentage:.2f}%")

if __name__ == '__main__':
    dataset = DataSet(balance_data=False)
    dataset.print_target_percentages()