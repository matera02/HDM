import torch
import torch.nn as nn
import numpy as np
from heartDisease import DataSet
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Creo un nuovo dataset con i dati sistemati per l'ANN
class ScaledDataSet:
    def __init__(self, dataset:DataSet):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(dataset.X_train)
        X_test_scaled = scaler.transform(dataset.X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        y_train = dataset.y_train.astype(np.int64)
        y_test = dataset.y_test.astype(np.int64)
        self.X_train =  X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

def get_skorch_ann_and_params(scaled_dataset:ScaledDataSet):
    input_size = scaled_dataset.X_train.shape[1]
    ann = NeuralNetClassifier(
        module=ANN,
        module__input_size=input_size,
        module__hidden_size=32,
        module__num_classes=2,
        module__dropout_rate=0.5,
        max_epochs=20,
        lr=0.01,
        batch_size=32,
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
    )
    ann_params = {
        'lr': [0.001, 0.01, 0.1],
        'max_epochs': [10, 20, 50, 100],
        'batch_size': [16, 32, 64],
        'module__hidden_size': [32, 64, 128],
        'module__dropout_rate': [0.3, 0.5, 0.7]
    }
    return ann, ann_params