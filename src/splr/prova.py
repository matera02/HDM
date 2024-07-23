import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection, metrics, tree
from sklearn.preprocessing import StandardScaler
import optuna
import xgboost as xgb
import joblib

class BayesianOptimization:
    def __init__(self, dataset: str):
        self.dataset = pd.read_csv(dataset)
        self.prepare_data()

    def prepare_data(self):
        # Tutti i valori della colonna 'num' devono essere numerici
        self.dataset['num'] = pd.to_numeric(self.dataset['num'], errors='coerce')
        
        # Rimuovo righe con valori mancanti
        self.dataset.dropna(inplace=True)
        
        # Separo le caratteristiche dall'etichetta
        self.X = self.dataset.drop("num", axis=1).values
        self.y = self.dataset['num'].apply(lambda x: 1 if x > 0 else 0).values

        # Standardizzo le caratteristiche
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
            algorithm='SAMME'  # Specifica l'algoritmo SAMME per evitare l'avviso
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

    def process_dataset(self):
        # Random Forest
        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(self.optimize_rf, n_trials=5)
        best_params_rf = study_rf.best_params
        print("Migliori parametri Random Forest:", best_params_rf)
        joblib.dump(best_params_rf, 'best_params_rf.pkl')

        # AdaBoost
        study_adaboost = optuna.create_study(direction='minimize')
        study_adaboost.optimize(self.optimize_adaboost, n_trials=5)
        best_params_adaboost = study_adaboost.best_params
        print("Migliori parametri AdaBoost:", best_params_adaboost)
        joblib.dump(best_params_adaboost, 'best_params_adaboost.pkl')

        # Decision Tree
        study_dt = optuna.create_study(direction='minimize')
        study_dt.optimize(self.optimize_dt, n_trials=5)
        best_params_dt = study_dt.best_params
        print("Migliori parametri Decision Tree:", best_params_dt)
        joblib.dump(best_params_dt, 'best_params_dt.pkl')

        # XGBoost
        study_xgboost = optuna.create_study(direction='minimize')
        study_xgboost.optimize(self.optimize_xgboost, n_trials=5)
        best_params_xgboost = study_xgboost.best_params
        print("Migliori parametri XGBoost:", best_params_xgboost)
        joblib.dump(best_params_xgboost, 'best_params_xgboost.pkl')

        # Eseguire il training finale e valutare i modelli
        self.train_and_evaluate_models(best_params_rf, best_params_adaboost, best_params_dt, best_params_xgboost)

    def train_and_evaluate_models(self, best_params_rf, best_params_adaboost, best_params_dt, best_params_xgboost):
        # Dividere il dataset in training e test set
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
