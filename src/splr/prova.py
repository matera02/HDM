import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection, metrics, tree, linear_model
from sklearn.preprocessing import StandardScaler
import optuna
import xgboost as xgb
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class ANNModel(nn.Module):
    def __init__(self, input_dim, n_layers, units, dropout_rate):
        super(ANNModel, self).__init__()
        layers = []
        in_features = input_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = units[i]
        
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DataSet:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_csv(dataset_path)
        self.prepare_data()

    def prepare_data(self):
        self.dataset['num'] = pd.to_numeric(self.dataset['num'], errors='coerce')
        self.dataset.dropna(inplace=True)
        self.X = self.dataset.drop("num", axis=1).values
        self.y = self.dataset['num'].apply(lambda x: 1 if x > 0 else 0).values
        self.X = StandardScaler().fit_transform(self.X)

class BayesianOptimization:
    def __init__(self, data: DataSet):
        self.data = data

    def optimize_rf(self, trial):
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        n_estimators = trial.suggest_int("n_estimators", 100, 1500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        max_features = trial.suggest_float("max_features", 0.01, 1.0)

        model = ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            criterion=criterion,
        )

        return self.cross_validate_model(model)

    def optimize_adaboost(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1)

        model = ensemble.AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm='SAMME'
        )
        return self.cross_validate_model(model)

    def optimize_dt(self, trial):
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

        model = tree.DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        return self.cross_validate_model(model)

    def optimize_xgboost(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        model = xgb.XGBClassifier(**params)
        return self.cross_validate_model(model)

    def optimize_ann(self, trial):
        n_layers = trial.suggest_int('n_layers', 1, 5)
        units = [trial.suggest_int(f'n_units_layer_{i}', 4, 128) for i in range(n_layers)]
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        model = ANNModel(input_dim=self.data.X.shape[1], n_layers=n_layers, units=units, dropout_rate=dropout_rate)
        return self.cross_validate_ann_model(model, trial)

    def optimize_lr(self, trial):
        C = trial.suggest_loguniform('C', 1e-5, 1e2)
        solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        model = linear_model.LogisticRegression(C=C, solver=solver, max_iter=1000)
        return self.cross_validate_model(model)

    def cross_validate_model(self, model):
        kf = model_selection.StratifiedKFold(n_splits=5)
        accuracies = []
        for train_idx, test_idx in kf.split(X=self.data.X, y=self.data.y):
            x_train, y_train = self.data.X[train_idx], self.data.y[train_idx]
            x_test, y_test = self.data.X[test_idx], self.data.y[test_idx]

            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            fold_accuracy = metrics.accuracy_score(y_test, preds)
            accuracies.append(fold_accuracy)

        return -1.0 * np.mean(accuracies)

    def cross_validate_ann_model(self, model, trial):
        kf = model_selection.StratifiedKFold(n_splits=5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-1))

        accuracies = []
        for train_idx, test_idx in kf.split(X=self.data.X, y=self.data.y):
            x_train, y_train = self.data.X[train_idx], self.data.y[train_idx]
            x_test, y_test = self.data.X[test_idx], self.data.y[test_idx]

            train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
            test_data = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            model.to(device)
            model.train()

            for epoch in range(100):  # Increased epochs for better training
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            all_preds = []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x).squeeze()
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())

            fold_accuracy = metrics.accuracy_score(y_test, all_preds)
            accuracies.append(fold_accuracy)

        return -1.0 * np.mean(accuracies)

    def process_dataset(self):
        # Random Forest
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(self.optimize_rf, n_trials=10, n_jobs=8)
        best_params_rf = study_rf.best_params
        print("Migliori parametri Random Forest:", best_params_rf)
        joblib.dump(best_params_rf, 'best_params_rf.pkl')

        # AdaBoost
        study_adaboost = optuna.create_study(direction='maximize')
        study_adaboost.optimize(self.optimize_adaboost, n_trials=10, n_jobs=8)
        best_params_adaboost = study_adaboost.best_params
        print("Migliori parametri AdaBoost:", best_params_adaboost)
        joblib.dump(best_params_adaboost, 'best_params_adaboost.pkl')

        # Decision Tree
        study_dt = optuna.create_study(direction='maximize')
        study_dt.optimize(self.optimize_dt, n_trials=10, n_jobs=8)
        best_params_dt = study_dt.best_params
        print("Migliori parametri Decision Tree:", best_params_dt)
        joblib.dump(best_params_dt, 'best_params_dt.pkl')

        # XGBoost
        study_xgboost = optuna.create_study(direction='maximize')
        study_xgboost.optimize(self.optimize_xgboost, n_trials=10, n_jobs=8)
        best_params_xgboost = study_xgboost.best_params
        print("Migliori parametri XGBoost:", best_params_xgboost)
        joblib.dump(best_params_xgboost, 'best_params_xgboost.pkl')

        # ANN
        study_ann = optuna.create_study(direction='maximize')
        study_ann.optimize(self.optimize_ann, n_trials=10, n_jobs=8)
        best_params_ann = study_ann.best_params
        print("Migliori parametri ANN:", best_params_ann)
        joblib.dump(best_params_ann, 'best_params_ann.pkl')

        # Logistic Regression
        study_lr = optuna.create_study(direction='maximize')
        study_lr.optimize(self.optimize_lr, n_trials=10, n_jobs=8)
        best_params_lr = study_lr.best_params
        print("Migliori parametri Logistic Regression:", best_params_lr)
        joblib.dump(best_params_lr, 'best_params_lr.pkl')

        # Eseguire il training finale e valutare i modelli
        self.train_and_evaluate_models(best_params_rf, best_params_adaboost, best_params_dt, best_params_xgboost, best_params_ann, best_params_lr)

    def train_and_evaluate_models(self, best_params_rf, best_params_adaboost, best_params_dt, best_params_xgboost, best_params_ann, best_params_lr):
        models = {
            "Random Forest": ensemble.RandomForestClassifier(**best_params_rf),
            "AdaBoost": ensemble.AdaBoostClassifier(**best_params_adaboost),
            "Decision Tree": tree.DecisionTreeClassifier(**best_params_dt),
            "XGBoost": xgb.XGBClassifier(**best_params_xgboost),
            "Logistic Regression": linear_model.LogisticRegression(**best_params_lr)
        }

        kf = model_selection.StratifiedKFold(n_splits=5)
        for name, model in models.items():
            accuracies, precisions, recalls, f1s, mccs = [], [], [], [], []
            for train_idx, test_idx in kf.split(X=self.data.X, y=self.data.y):
                x_train, y_train = self.data.X[train_idx], self.data.y[train_idx]
                x_test, y_test = self.data.X[test_idx], self.data.y[test_idx]

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

                accuracies.append(metrics.accuracy_score(y_test, y_pred))
                precisions.append(metrics.precision_score(y_test, y_pred, average='weighted'))
                recalls.append(metrics.recall_score(y_test, y_pred, average='weighted'))
                f1s.append(metrics.f1_score(y_test, y_pred, average='weighted'))
                mccs.append(metrics.matthews_corrcoef(y_test, y_pred))
                self.plot_roc_curve(y_test, y_prob, name)

            print(f"{name} - Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1 Score: {np.mean(f1s):.4f}, MCC: {np.mean(mccs):.4f}")

        # Modello ANN
        ann_model = self.build_ann_model(best_params_ann)
        self.train_ann_model(ann_model, kf, best_params_ann)

    def build_ann_model(self, best_params_ann):
        n_layers = best_params_ann['n_layers']
        units = [best_params_ann[f'n_units_layer_{i}'] for i in range(n_layers)]
        dropout_rate = best_params_ann['dropout_rate']
        
        model = ANNModel(input_dim=self.data.X.shape[1], n_layers=n_layers, units=units, dropout_rate=dropout_rate)
        return model

    def train_ann_model(self, model, kf, best_params_ann):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params_ann['lr'])

        accuracies, precisions, recalls, f1s, mccs = [], [], [], [], []
        for train_idx, test_idx in kf.split(X=self.data.X, y=self.data.y):
            x_train, y_train = self.data.X[train_idx], self.data.y[train_idx]
            x_test, y_test = self.data.X[test_idx], self.data.y[test_idx]

            train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
            test_data = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            model.train()
            for epoch in range(100):  # Increased epochs for better training
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x).squeeze()
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())

            accuracies.append(metrics.accuracy_score(all_labels, all_preds))
            precisions.append(metrics.precision_score(all_labels, all_preds, average='weighted'))
            recalls.append(metrics.recall_score(all_labels, all_preds, average='weighted'))
            f1s.append(metrics.f1_score(all_labels, all_preds, average='weighted'))
            mccs.append(metrics.matthews_corrcoef(all_labels, all_preds))

        print(f"ANN - Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1 Score: {np.mean(f1s):.4f}, MCC: {np.mean(mccs):.4f}")

    def plot_roc_curve(self, y_test, y_prob, model_name):
        fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_learning_curves(self, model, X, y, model_name):
        train_sizes, train_scores, test_scores = model_selection.learning_curve(model, X, y, cv=5, scoring='accuracy')
        train_errors = 1 - train_scores
        test_errors = 1 - test_scores
        train_errors_std = np.std(train_errors, axis=1)
        test_errors_std = np.std(test_errors, axis=1)
        train_errors_var = np.var(train_errors, axis=1)
        test_errors_var = np.var(test_errors, axis=1)

        print(
            f"{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}"
        )

        mean_train_errors = 1 - np.mean(train_scores, axis=1)
        mean_test_errors = 1 - np.mean(test_scores, axis=1)

        plt.figure(figsize=(16, 10))
        plt.plot(train_sizes, mean_train_errors, label='Training error', color='green')
        plt.plot(train_sizes, mean_test_errors, label='Testing error', color='red')
        plt.title(f'Learning curve for {model_name}')
        plt.xlabel('Training set size')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    dataset = DataSet(dataset_path='src/splr/data/processedDataset/heart_disease.csv')
    bo = BayesianOptimization(data=dataset)
    bo.process_dataset()
