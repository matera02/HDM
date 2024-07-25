from heartDisease import DataSet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import optuna

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

class ANNModel:
    def __init__(self, dataset, n_trials=1000, n_jobs=8):
        self.dataset = dataset
        self.input_size = self.dataset.X.shape[1]
        self.num_classes = len(np.unique(self.dataset.y))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        # Conversione dei dati in tensori PyTorch
        self.X = torch.FloatTensor(self.dataset.X).to(self.device)
        self.y = torch.LongTensor(self.dataset.y).to(self.device)
    
    def train_evaluate(self, trial):
        # Definizione degli iperparametri da ottimizzare
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        epochs = trial.suggest_int('epochs', 10, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        # K-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            X_train_fold, X_val_fold = self.X[train_idx], self.X[val_idx]
            y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]

            model = ANN(self.input_size, hidden_size, self.num_classes, dropout_rate).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            train_dataset = TensorDataset(X_train_fold, y_train_fold)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                outputs = model(X_val_fold)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_val_fold).sum().item() / y_val_fold.size(0)
                fold_accuracies.append(accuracy)
        
        return np.mean(fold_accuracies)
    
    def optimize_hyperparameters(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.train_evaluate, n_trials=self.n_trials, n_jobs=self.n_jobs)
        return study.best_params, study.best_value
    
    def run(self):
        print("Ottimizzazione degli iperparametri...")
        best_params, best_accuracy = self.optimize_hyperparameters()
        print("Migliori parametri trovati:", best_params)
        print(f"Miglior accuratezza media (cross-validation): {best_accuracy:.4f}")
        
        # Addestramento del modello finale con i migliori parametri
        best_model = ANN(self.input_size, best_params['hidden_size'], self.num_classes, best_params['dropout_rate']).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
        
        train_dataset = TensorDataset(self.X, self.y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        
        for epoch in range(best_params['epochs']):
            for inputs, labels in train_loader:
                outputs = best_model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Valutazione finale
        best_model.eval()
        with torch.no_grad():
            outputs = best_model(self.X)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(self.y.cpu().numpy(), predicted.cpu().numpy())
        
        print(f"Accuratezza finale: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y.cpu().numpy(), predicted.cpu().numpy()))
        
        # Plot della curva di apprendimento
        self.plot_learning_curve(best_model, criterion, optimizer, best_params['batch_size'])

    
    def plot_learning_curve(self, model, criterion, optimizer, batch_size):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        # Dividiamo il dataset in train e test solo per il plotting
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(100):  # Usiamo 100 epoche per la curva di apprendimento
            model.train()
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                train_loss = criterion(train_outputs, y_train)
                _, train_predicted = torch.max(train_outputs.data, 1)
                train_accuracy = (train_predicted == y_train).sum().item() / y_train.size(0)
            
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_accuracy = (test_predicted == y_test).sum().item() / y_test.size(0)
        
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
    
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Learning Curve - Loss')
    
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Learning Curve - Accuracy')
    
        plt.tight_layout()
        plt.show()

if __name__  == '__main__':
    # Uso della classe
    dataset = DataSet()
    ann_model = ANNModel(dataset, n_trials=20)
    ann_model.run()