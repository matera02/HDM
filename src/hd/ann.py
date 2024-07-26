import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, learning_curve
import optuna
import json
from heartDisease import DataSet
from src.util.utility import PyTorchWrapper
import os


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
    def __init__(self, dataset, n_trials=100, n_jobs=8):
        self.dataset = dataset
        self.input_size = self.dataset.X.shape[1]
        self.num_classes = len(np.unique(self.dataset.y))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        self.X = torch.FloatTensor(self.dataset.X).to(self.device)
        self.y = torch.LongTensor(self.dataset.y).to(self.device)
    
    def train_evaluate(self, trial):
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        epochs = trial.suggest_int('epochs', 10, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

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
    
    def analyze_overfitting(self, train_scores, test_scores):
        train_mean = np.mean(train_scores[-1])
        test_mean = np.mean(test_scores[-1])
        train_std = np.std(train_scores[-1])
        test_std = np.std(test_scores[-1])
        gap = train_mean - test_mean
        
        analysis = []
        if gap > 0.1:
            analysis.append(f"Possibile overfitting. Gap tra train and test: {gap:.3f}")
        elif train_mean < 0.8:
            analysis.append(f"Possibile underfitting. Train score: {train_mean:.3f}")
        else:
            analysis.append("Il modello sembra ben bilanciato.")
    
        analysis.append(f"Train score: media = {train_mean:.3f}, std = {train_std:.3f}")
        analysis.append(f"Test score: media = {test_mean:.3f}, std = {test_std:.3f}")
    
        train_variance = np.var(train_scores[-1])
        test_variance = np.var(test_scores[-1])
        analysis.append(f"Train varianza: {train_variance:.3f}")
        analysis.append(f"Test varianza: {test_variance:.3f}")
        
        if test_std > 0.1:
            analysis.append("Il modello mostra una certa instabilità sui dati di test.")
        else:
            analysis.append("Il modello sembra stabile sui dati di test.")
        return "\n".join(analysis)

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), savefig = 'src/hd/data/results/ann/learning_curve_ann.png'):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.savefig(savefig)
        plt.show()
        return train_scores, test_scores

    def run(self, filename = 'ann_model_info.json', dir='src/hd/data/results/ann'):
        print("Optimizing hyperparameters...")
        best_params, best_accuracy = self.optimize_hyperparameters()
        print("Best parameters found:", best_params)
        print(f"Best mean accuracy (cross-validation): {best_accuracy:.4f}")
        
        best_model = ANN(self.input_size, best_params['hidden_size'], self.num_classes, best_params['dropout_rate']).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
        
        sklearn_model = PyTorchWrapper(
            best_model, criterion, optimizer, 
            best_params['batch_size'], best_params['epochs'], self.device
        )
        
        X_np = self.X.cpu().numpy()
        y_np = self.y.cpu().numpy()
        
        sklearn_model.fit(X_np, y_np)
        
        print("Plotting learning curve...")
        train_scores, test_scores = self.plot_learning_curve(
            sklearn_model, "Learning Curve (ANN)", X_np, y_np, ylim=(0.7, 1.01), cv=5, n_jobs=-1)

        predictions = sklearn_model.predict(X_np)
        accuracy = accuracy_score(y_np, predictions)
        precision = precision_score(y_np, predictions, average='weighted')
        recall = recall_score(y_np, predictions, average='weighted')
        f1 = f1_score(y_np, predictions, average='weighted')
        
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_np, predictions))
        
        overfitting_analysis = self.analyze_overfitting(train_scores, test_scores)
        print("Overfitting Analysis:")
        print(overfitting_analysis)

        self.plot_score_distribution(train_scores, test_scores)

        model_info = {
            "model_name": "ANN",
            "best_params": best_params,
            "best_accuracy": best_accuracy,
            "final_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "overfitting_analysis": overfitting_analysis
        }
        # Viene creata la cartella di destinazione se non esiste
        os.makedirs(dir, exist_ok=True)
        dest = os.path.join(dir, filename)
        with open(dest, "w") as f:
            json.dump(model_info, f, indent=4)

    def plot_score_distribution(self, train_scores, test_scores, savefig='src/hd/data/results/ann/score_distribution_ann.png'):
        plt.figure(figsize=(10, 6))
        plt.title(f"Distribuzione degli score - ANN")
        plt.boxplot([train_scores[-1], test_scores[-1]], labels=['Train', 'Test'])
        plt.ylabel("Score")
        plt.savefig(savefig)
        plt.show()


if __name__  == '__main__':
    dataset = DataSet()
    ann_model = ANNModel(dataset, n_trials=10)
    ann_model.run()