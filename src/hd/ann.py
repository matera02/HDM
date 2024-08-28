import torch.nn as nn
import numpy as np
from heartDisease import DataSet

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

# serve per ANN
class ANNDataSet:
    def __init__(self, dataset: DataSet):
        self.X_train = dataset.X_train.astype(np.float32)
        self.X_test = dataset.X_test.astype(np.float32)
        self.y_train = dataset.y_train.astype(np.int64)
        self.y_test = dataset.y_test.astype(np.int64)