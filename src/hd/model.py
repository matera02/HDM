import numpy as np
import matplotlib.pyplot as plt
import json
import os
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from skorch import NeuralNetClassifier
from joblib import Parallel, delayed
from src.hd.heartDisease import DataSet
from src.hd.ann import ANN, ANNDataSet
from sklearn.model_selection import cross_validate


class Model:
    def __init__(self, dataset, model_class, model_name, n_trials=100, cv=5, n_jobs=4):
        self.dataset = dataset
        self.model_class = model_class
        self.model_name = model_name
        self.n_trials = n_trials
        self.cv = cv
        self.n_jobs = n_jobs

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), savefig=''):
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
        plt.close()
        return train_scores, test_scores

    def analyze_overfitting(self, train_scores, test_scores):
        train_mean = np.mean(train_scores[-1])
        test_mean = np.mean(test_scores[-1])
        train_std = np.std(train_scores[-1])
        test_std = np.std(test_scores[-1])
        gap = train_mean - test_mean
        
        analysis = []
        if gap > 0.1:
            analysis.append(f"Possibile overfitting. Gap tra train e test: {gap:.6f}")
        elif train_mean < 0.8:
            analysis.append(f"Possibile underfitting. Train score: {train_mean:.6f}")
        else:
            analysis.append("Il modello sembra ben bilanciato.")
    
        analysis.append(f"Train score: media = {train_mean:.6f}, std = {train_std:.6f}")
        analysis.append(f"Test score: media = {test_mean:.6f}, std = {test_std:.6f}")
    
        train_variance = np.var(train_scores[-1])
        test_variance = np.var(test_scores[-1])
        analysis.append(f"Train varianza: {train_variance:.6f}")
        analysis.append(f"Test varianza: {test_variance:.6f}")
        
        if test_std > 0.1:
            analysis.append("Il modello mostra una certa instabilità sui dati di test.")
        else:
            analysis.append("Il modello sembra stabile sui dati di test.")
        return "\n".join(analysis)

    def objective(self, trial):
        match self.model_name:
            case "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 10, 200),
                    max_depth=trial.suggest_int('max_depth', 1, 32),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=1  # Usiamo 1 qui perché stiamo già parallelizzando a livello superiore
                )
            case "Gradient Boosting":
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 10, 200),
                    learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
                    max_depth=trial.suggest_int('max_depth', 1, 32),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                    random_state=42
                )
            case "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=trial.suggest_int('max_depth', 1, 32),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                    random_state=42,
                    class_weight='balanced'
                )
            case "Logistic Regression":
                model = LogisticRegression(
                    C=trial.suggest_float('C', 1e-5, 1e5, log=True),
                    penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
                    solver='liblinear',
                    random_state=42,
                    class_weight='balanced',
                    max_iter=trial.suggest_int("max_iter", 100, 1000)
                )
            case "ANN":
                model = NeuralNetClassifier(
                    module=ANN,
                    module__input_size=self.dataset.X_train.shape[1],
                    module__hidden_size=trial.suggest_categorical('module__hidden_size', [32, 64, 128]),
                    module__num_classes=2,
                    module__dropout_rate=trial.suggest_float('module__dropout_rate', 0.1, 0.7),
                    max_epochs=trial.suggest_int('max_epochs', 10, 100),
                    lr=trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                    batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                    iterator_train__shuffle=True,
                    optimizer=torch.optim.Adam,
                    criterion=nn.CrossEntropyLoss,
                )
            case _:
                raise ValueError(f"Modello sconosciuto: {self.model_name}")

        scores = cross_val_score(model, self.dataset.X_train, self.dataset.y_train, cv=self.cv, n_jobs=1)
        return scores.mean()


    def run(self, dir_json, savefig_learning_curve):
        print(f"Sto eseguendo l'ottimizzazione con Optuna per {self.model_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        print(f"Migliori parametri per {self.model_name}:", study.best_params)
        
        match self.model_name:
            case "Random Forest":
                best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=self.n_jobs)
            case "Gradient Boosting":
                best_model = GradientBoostingClassifier(**study.best_params, random_state=42)
            case "Decision Tree":
                best_model = DecisionTreeClassifier(**study.best_params, random_state=42)
            case "Logistic Regression":
                best_model = LogisticRegression(**study.best_params, random_state=42)
            case "ANN":
                best_model = NeuralNetClassifier(
                module=ANN,
                module__input_size=self.dataset.X_train.shape[1],
                module__num_classes=2,
                **study.best_params,
                iterator_train__shuffle=True,
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
            )
            case _:
                raise ValueError(f"Modello sconosciuto: {self.model_name}")


        best_model.fit(self.dataset.X_train, self.dataset.y_train)
        model_pred = best_model.predict(self.dataset.X_test)

        accuracy = accuracy_score(self.dataset.y_test, model_pred)
        precision = precision_score(self.dataset.y_test, model_pred, average='weighted')
        recall = recall_score(self.dataset.y_test, model_pred, average='weighted')
        f1 = f1_score(self.dataset.y_test, model_pred, average='weighted')

        print(f"{self.model_name} Accuracy:", accuracy)
        print(f"{self.model_name} Precision:", precision)
        print(f"{self.model_name} Recall:", recall)
        print(f"{self.model_name} F1 Score:", f1)
        print(f"{self.model_name} Classification Report:")
        print(classification_report(self.dataset.y_test, model_pred))

        print(f"Plot della curva di apprendimento {self.model_name}...")
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, self.dataset.X_train, self.dataset.y_train, 
            cv=self.cv, n_jobs=self.n_jobs, train_sizes=np.linspace(.1, 1.0, 5)
        )
        self.plot_learning_curve(best_model, f"Learning Curve - {self.model_name}", 
                                 self.dataset.X_train, self.dataset.y_train, 
                                 ylim=(0.4, 1.01), cv=self.cv, n_jobs=self.n_jobs, savefig=savefig_learning_curve)

        print(f"{self.model_name} Overfitting Analysis: ")
        overfitting_analysis = self.analyze_overfitting(train_scores, test_scores)
        print(overfitting_analysis)

        model_info = {
            "model_name": self.model_name,
            "best_params": study.best_params,
            "best_accuracy": study.best_value,
            "final_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "overfitting_analysis": overfitting_analysis
        }
        os.makedirs(dir_json, exist_ok=True)
        filename = f"{self.model_name.lower().replace(' ', '_')}_model_info.json"
        dest = os.path.join(dir_json, filename)
        with open(dest, "w") as f:
            json.dump(model_info, f, indent=4)

def run_model(model_class, model_name, dataset, dir_json, savefig_learning_curve):
    model = Model(dataset, model_class, model_name, n_jobs=4)
    model.run(dir_json, savefig_learning_curve)


def plot_complexity_curve(filename, savefig, model_name, complexity_param, param_range, X, y, cv=5, scoring='accuracy'):
    with open(filename, 'r') as f:
        model_info = json.load(f)
    
    best_params = model_info['best_params']
    
    train_scores = []
    test_scores = []
    
    for param_value in param_range:
        params = best_params.copy()
        params[complexity_param] = param_value
        
        match model_name:
            case "Decision Tree":
                model = DecisionTreeClassifier(**params, random_state=42)
            case "Random Forest":
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            case "Logistic Regression":
                model = LogisticRegression(**params, random_state=42)
            case "Gradient Boosting":
                model = GradientBoostingClassifier(**params, random_state=42)
            case "ANN":
                model = NeuralNetClassifier(
                module=ANN,
                module__input_size=X.shape[1],
                module__num_classes=2,
                **params,
                iterator_train__shuffle=True,
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
            )
            case _:
                raise ValueError(f"Unknown model name: {model_name}")

        
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        
        train_scores.append(np.mean(scores['train_score']))
        test_scores.append(np.mean(scores['test_score']))
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Complexity Curve - {model_name}")
    plt.xlabel(complexity_param)
    plt.ylabel("Score")
    
    if model_name == "Logistic Regression" and complexity_param == 'C':
        plt.semilogx(param_range, train_scores, 'o-', color="r", label="Training score")
        plt.semilogx(param_range, test_scores, 'o-', color="g", label="Test score")
        plt.xscale('log')
    else:
        plt.plot(param_range, train_scores, 'o-', color="r", label="Training score")
        plt.plot(param_range, test_scores, 'o-', color="g", label="Test score")
    
    plt.fill_between(param_range, train_scores, test_scores, alpha=0.1, color="g")
    plt.legend(loc="best")
    plt.grid()
    
    plt.savefig(savefig)
    plt.close()

if __name__ == '__main__':
    dataset = DataSet(balance_data=True)

    ann_dataset = ANNDataSet(dataset) # per ANN
    
    plot_complexity_curve(
        filename='src/hd/data/results/lr/logistic_regression_model_info.json',
        savefig='src/hd/data/results/lr/complexity_curve_lr.png',
        model_name="Logistic Regression", 
        complexity_param="C", 
        param_range=np.logspace(-5, 5, 11),  # Varieremo C da 1e-5 a 1e5, su scala logaritmica
        X=dataset.X_train, 
        y=dataset.y_train
    )
    

"""
(LogisticRegression, 'Logistic Regression', dataset, 'src/hd/data/results/lr/', 'src/hd/data/results/lr/learning_curve_lr.png')
    
    lr_model = Model(dataset=dataset,model_class=LogisticRegression, model_name="Logistic Regression")
    lr_model.run(
        dir_json='src/hd/data/results/lr/',
        savefig_learning_curve='src/hd/data/results/lr/learning_curve_lr.png'
    )
# Gradient Boosting
    plot_complexity_curve(
        filename='src/hd/data/results/gb/gradient_boosting_model_info.json',
        savefig='src/hd/data/results/gb/complexity_curve_gb.png',
        model_name="Gradient Boosting", 
        complexity_param="n_estimators", 
        param_range=range(10, 251, 5),  # Varieremo n_estimators da 10 a 200, con step di 10
        X=dataset.X_train, 
        y=dataset.y_train
    )


# Random Forest
    plot_complexity_curve(
        filename='src/hd/data/results/rf/random_forest_model_info.json',
        savefig='src/hd/data/results/rf/complexity_curve_rf.png',
        model_name="Random Forest", 
        complexity_param="n_estimators", 
        param_range=range(10, 201, 3),  # Varieremo n_estimators da 10 a 200, con step di 10
        X=dataset.X_train, 
        y=dataset.y_train
    )


# Gradient Boosting
    plot_complexity_curve(
        filename='src/hd/data/results/gb/gradient_boosting_model_info.json',
        savefig='src/hd/data/results/gb/complexity_curve_gb_max_depth.png',
        model_name="Gradient Boosting", 
        complexity_param="max_depth", 
        param_range=range(1, 33), 
        X=dataset.X_train, 
        y=dataset.y_train
    )


# Per Decision Tree, usiamo min_samples_split 
    plot_complexity_curve(
        filename= 'src/hd/data/results/dt/decision_tree_model_info.json',
        savefig='src/hd/data/results/dt/complexity_curve_dt_min_samples_split.png',
        model_name="Decision Tree", 
        complexity_param="min_samples_split", 
        param_range=range(2, 51),  # a quanto pare 20 è il miglior parametro
        X=dataset.X_train, 
        y=dataset.y_train
    )



# Per Decision Tree, usiamo min_samples_leaf 
    plot_complexity_curve(
        filename= 'src/hd/data/results/dt/decision_tree_model_info.json',
        savefig='src/hd/data/results/dt/complexity_curve_dt_min_samples_leaf.png',
        model_name="Decision Tree", 
        complexity_param="min_samples_leaf", 
        param_range=range(2, 51),  # a quanto pare 20 è il miglior parametro
        X=dataset.X_train, 
        y=dataset.y_train
    )



# Per Decision Tree, usiamo min_samples_split 
    plot_complexity_curve(
        filename= 'src/hd/data/results/dt/decision_tree_model_info.json',
        savefig='src/hd/data/results/dt/complexity_curve_dt_min_samples_split.png',
        model_name="Decision Tree", 
        complexity_param="min_samples_split", 
        param_range=range(2, 51),  # a quanto pare 20 è il miglior parametro
        X=dataset.X_train, 
        y=dataset.y_train
    )



# Random Forest
    plot_complexity_curve(
        filename='src/hd/data/results/rf/random_forest_model_info.json',
        savefig='src/hd/data/results/rf/complexity_curve_rf_max_depth.png',
        model_name="Random Forest", 
        complexity_param="max_depth", 
        param_range=range(1, 33),
        X=dataset.X_train, 
        y=dataset.y_train
    )



# Logistic Regression
    plot_complexity_curve(
        filename='src/hd/data/results/lr/logistic_regression_model_info.json',
        savefig='src/hd/data/results/lr/complexity_curve_lr.png',
        model_name="Logistic Regression", 
        complexity_param="C", 
        param_range=np.logspace(-5, 5, 11),  # Varieremo C da 1e-5 a 1e5, su scala logaritmica
        X=dataset.X_train, 
        y=dataset.y_train
    )

 # Per Decision Tree, usiamo max_depth come misura di complessità
    plot_complexity_curve(
        filename= 'src/hd/data/results/dt/decision_tree_model_info.json',
        savefig='src/hd/data/results/dt/complexity_curve_dt.png',
        model_name="Decision Tree", 
        complexity_param="max_depth", 
        param_range=range(1, 33),  # Varieremo max_depth da 1 a 32
        X=dataset.X_train, 
        y=dataset.y_train
    )

     # Random Forest
    plot_complexity_curve(
        filename='src/hd/data/results/rf/random_forest_model_info.json',
        savefig='src/hd/data/results/rf/complexity_curve_rf.png',
        model_name="Random Forest", 
        complexity_param="n_estimators", 
        param_range=range(10, 201, 10),  # Varieremo n_estimators da 10 a 200, con step di 10
        X=dataset.X_train, 
        y=dataset.y_train
    )
   """

    
"""
# Gradient Boosting
    plot_complexity_curve(
        filename='src/hd/data/results/gb/gradient_boosting_model_info.json',
        savefig='src/hd/data/results/gb/complexity_curve_gb.png',
        model_name="Gradient Boosting", 
        complexity_param="n_estimators", 
        param_range=range(10, 201, 10),  # Varieremo n_estimators da 10 a 200, con step di 10
        X=dataset.X_train, 
        y=dataset.y_train
    )

    # ANN
    plot_complexity_curve(
        filename='src/hd/data/results/ann/ann_model_info.json',
        savefig='src/hd/data/results/ann/complexity_curve_ann.png',
        model_name="ANN", 
        complexity_param="max_epochs", 
        param_range=range(10, 101, 10),  # Varieremo max_epochs da 10 a 100, con step di 10
        X=ann_dataset.X_train, 
        y=ann_dataset.y_train
    )

    plot_complexity_curve(
        filename='src/hd/data/results/ann/ann_model_info.json',
        savefig='src/hd/data/results/ann/complexity_curve_ann_module__hidden_size.png',
        model_name="ANN", 
        complexity_param="module__hidden_size", 
        param_range=[2, 4, 8, 16, 32, 64, 128],
        X=ann_dataset.X_train, 
        y=ann_dataset.y_train
    )
"""
    

    

"""
models = [
        (DecisionTreeClassifier, "Decision Tree", dataset, 'src/hd/data/results/dt/', 'src/hd/data/results/dt/learning_curve_dt.png'),
        (LogisticRegression, 'Logistic Regression', dataset, 'src/hd/data/results/lr/', 'src/hd/data/results/lr/learning_curve_lr.png'),
        (RandomForestClassifier, 'Random Forest', dataset, 'src/hd/data/results/rf/', 'src/hd/data/results/rf/learning_curve_rf.png'),
        (GradientBoostingClassifier, 'Gradient Boosting', dataset, 'src/hd/data/results/gb/', 'src/hd/data/results/gb/learning_curve_gb.png'),
        (NeuralNetClassifier, 'ANN', ann_dataset, 'src/hd/data/results/ann/', 'src/hd/data/results/ann/learning_curve_ann.png')
    ]

    Parallel(n_jobs=4)(delayed(run_model)(*args) for args in models)
"""
    
    