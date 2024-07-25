import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self, dataset_path='src/hd/data/processedDataset/heart_disease.csv', test_size=0.2):
        self.dataset = pd.read_csv(dataset_path)
        self.prepare_data()
        self.calculate_target_percentages()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.X_train, self.X_test = self.scale_features()


    def prepare_data(self):
        self.dataset['num'] = pd.to_numeric(self.dataset['num'], errors='coerce')
        self.dataset.dropna(inplace=True)
        self.X = self.dataset.drop("num", axis=1).values
        self.y = self.dataset['num'].apply(lambda x: 1 if x > 0 else 0).values
        self.X = StandardScaler().fit_transform(self.X)

    def calculate_target_percentages(self):
        unique, counts = np.unique(self.y, return_counts=True)
        total = len(self.y)
        self.target_percentages = {value: (count / total) * 100 for value, count in zip(unique, counts)}

    def print_target_percentages(self):
        print("Percentuali della variabile target:")
        for value, percentage in self.target_percentages.items():
            print(f"Classe {value}: {percentage:.2f}%")

    def scale_features(self):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

if __name__ == '__main__':
    dataset =  DataSet()
    dataset.print_target_percentages()
