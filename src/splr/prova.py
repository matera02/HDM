import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection, metrics, tree
from sklearn.preprocessing import StandardScaler
import optuna
import xgboost as xgb
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

class BayesianOptimization:
    def __init__(self, dataset: str):
        self.dataset = pd.read_csv(dataset)
        self.prepare_data()

    def prepare_data(self):
        self.dataset['num'] = pd.to_numeric(self.dataset['num'], errors='coerce')
        self.dataset.dropna(inplace=True)
        self.X = self.dataset.drop("num", axis=1).values
        self.y = self.dataset['num'].apply(lambda x: 1 if x > 0 else 0).values
        self.X = StandardScaler().fit_transform(self.X)

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
        
        model = ANNModel(input_dim=self.X.shape[1], n_layers=n_layers, units=units, dropout_rate=dropout_rate)
        return self.cross_validate_ann_model(model, trial)

    def cross_validate_model(self, model):
        kf = model_selection.StratifiedKFold(n_splits=5)
        accuracies = []
        for train_idx, test_idx in kf.split(X=self.X, y=self.y):
            x_train, y_train = self.X[train_idx], self.y[train_idx]
            x_test, y_test = self.X[test_idx], self.y[test_idx]

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
        for train_idx, test_idx in kf.split(X=self.X, y=self.y):
            x_train, y_train = self.X[train_idx], self.y[train_idx]
            x_test, y_test = self.X[test_idx], self.y[test_idx]

            train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
            test_data = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            model.to(device)
            model.train()

            for epoch in range(20):  # Increased epochs for better training
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
        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(self.optimize_rf, n_trials=50, n_jobs=8)
        best_params_rf = study_rf.best_params
        print("Migliori parametri Random Forest:", best_params_rf)
        joblib.dump(best_params_rf, 'best_params_rf.pkl')

        # AdaBoost
        study_adaboost = optuna.create_study(direction='minimize')
        study_adaboost.optimize(self.optimize_adaboost, n_trials=50, n_jobs=8)
        best_params_adaboost = study_adaboost.best_params
        print("Migliori parametri AdaBoost:", best_params_adaboost)
        joblib.dump(best_params_adaboost, 'best_params_adaboost.pkl')

        # Decision Tree
        study_dt = optuna.create_study(direction='minimize')
        study_dt.optimize(self.optimize_dt, n_trials=50, n_jobs=8)
        best_params_dt = study_dt.best_params
        print("Migliori parametri Decision Tree:", best_params_dt)
        joblib.dump(best_params_dt, 'best_params_dt.pkl')

        # XGBoost
        study_xgboost = optuna.create_study(direction='minimize')
        study_xgboost.optimize(self.optimize_xgboost, n_trials=50, n_jobs=8)
        best_params_xgboost = study_xgboost.best_params
        print("Migliori parametri XGBoost:", best_params_xgboost)
        joblib.dump(best_params_xgboost, 'best_params_xgboost.pkl')

        # ANN
        study_ann = optuna.create_study(direction='minimize')
        study_ann.optimize(self.optimize_ann, n_trials=50, n_jobs=8)
        best_params_ann = study_ann.best_params
        print("Migliori parametri ANN:", best_params_ann)
        joblib.dump(best_params_ann, 'best_params_ann.pkl')

        # Eseguire il training finale e valutare i modelli
        self.train_and_evaluate_models(best_params_rf, best_params_adaboost, best_params_dt, best_params_xgboost, best_params_ann)

    def train_and_evaluate_models(self, best_params_rf, best_params_adaboost, best_params_dt, best_params_xgboost, best_params_ann):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)

        # Modello Random Forest
        rf = ensemble.RandomForestClassifier(**best_params_rf)
        rf.fit(x_train, y_train)
        self.evaluate_model(rf, x_test, y_test, "Random Forest")

        # Modello AdaBoost
        adaboost = ensemble.AdaBoostClassifier(**best_params_adaboost)
        adaboost.fit(x_train, y_train)
        self.evaluate_model(adaboost, x_test, y_test, "AdaBoost")

        # Modello Decision Tree
        dt = tree.DecisionTreeClassifier(**best_params_dt)
        dt.fit(x_train, y_train)
        self.evaluate_model(dt, x_test, y_test, "Decision Tree")

        # Modello XGBoost
        xgb_model = xgb.XGBClassifier(**best_params_xgboost)
        xgb_model.fit(x_train, y_train)
        self.evaluate_model(xgb_model, x_test, y_test, "XGBoost")

        # Modello ANN
        ann_model = self.build_ann_model(best_params_ann)
        self.train_ann_model(ann_model, x_train, y_train, x_test, y_test, best_params_ann)

    def build_ann_model(self, best_params_ann):
        n_layers = best_params_ann['n_layers']
        units = [best_params_ann[f'n_units_layer_{i}'] for i in range(n_layers)]
        dropout_rate = best_params_ann['dropout_rate']
        
        model = ANNModel(input_dim=self.X.shape[1], n_layers=n_layers, units=units, dropout_rate=dropout_rate)
        return model

    def train_ann_model(self, model, x_train, y_train, x_test, y_test, best_params_ann):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params_ann['lr'])

        train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_data = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        model.train()
        for epoch in range(20):  # Increased epochs for better training
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self.evaluate_ann_model(model, test_loader)

    def evaluate_ann_model(self, model, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = metrics.accuracy_score(all_labels, all_preds)
        precision = metrics.precision_score(all_labels, all_preds, average='weighted')
        recall = metrics.recall_score(all_labels, all_preds, average='weighted')
        f1 = metrics.f1_score(all_labels, all_preds, average='weighted')
        print(f"ANN - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    def evaluate_model(self, model, x_test, y_test, model_name):
        y_pred = model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    bo = BayesianOptimization(dataset='src/splr/data/processedDataset/heart_disease.csv')
    bo.process_dataset()
