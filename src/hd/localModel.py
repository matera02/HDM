from src.hd.model import Model
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
class LocalModel(Model):
    def __init__(self, dataset, model_class, model_name, **params):
        super().__init__(dataset, model_class, model_name)

    def run(self, save_json, savefig_learning_curve, **params):
        match self.model_name:
            case "Random Forest":
                best_model = RandomForestClassifier(**params, random_state=42, n_jobs=self.n_jobs)
            case "Gradient Boosting":
                best_model = GradientBoostingClassifier(**params, random_state=42)
            case "Decision Tree":
                best_model = DecisionTreeClassifier(**params, random_state=42)
            case "Logistic Regression":
                best_model = LogisticRegression(**params, random_state=42)
            case "ANN":
                best_model = NeuralNetClassifier(
                module=ANN,
                module__input_size=self.dataset.X_train.shape[1],
                module__num_classes=2,
                **params,
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
            "params": params,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "overfitting_analysis": overfitting_analysis
        }

        with open(save_json, "w") as f:
            json.dump(model_info, f, indent=4)


if __name__ == '__main__':
    dataset = DataSet(balance_data=True)

    ann_dataset = ANNDataSet(dataset)


    lr_params = {
        "C": 1e2,
        "penalty": "l1",
        "solver": "liblinear"
    }
    
    local_lr = LocalModel(
        dataset=dataset,
        model_class=LogisticRegression,
        model_name="Logistic Regression"
    )

    local_lr.run(
        save_json="src/hd/data/results/lr/local_lr_model_info.json",
        savefig_learning_curve="src/hd/data/results/lr/learning_curve_local_lr_model.png",
        **lr_params
    )


"""
# impostando a 10 otteniamo gli stessi risultati
    dt_params = {
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 7
    }

    local_dt = LocalModel(
        dataset=dataset,
        model_class=DecisionTreeClassifier,
        model_name="Decision Tree"
    )

    local_dt.run(
        save_json="src/hd/data/results/dt/local_dt_model_info.json",
        savefig_learning_curve="src/hd/data/results/dt/learning_curve_local_dt_model.png",
        **dt_params
    )



rf_params = {
        "n_estimators": 97,
        "max_depth": 9,
        "min_samples_split": 16,
        "min_samples_leaf": 3
    }

    local_rf = LocalModel(
        dataset=dataset,
        model_class=RandomForestClassifier,
        model_name="Random Forest"
    )

    local_rf.run(
        save_json="src/hd/data/results/rf/local_rf_model_info.json",
        savefig_learning_curve="src/hd/data/results/rf/learning_curve_local_rf_model.png",
        **rf_params
    )




lr_params = {
        "C": 1e2,
        "penalty": "l1",
        "solver": "saga"
    }
    
    local_lr = LocalModel(
        dataset=dataset,
        model_class=LogisticRegression,
        model_name="Logistic Regression"
    )

    local_lr.run(
        save_json="src/hd/data/results/lr/local_lr_model_info.json",
        savefig_learning_curve="src/hd/data/results/lr/learning_curve_local_lr_model.png",
        **lr_params
    )


# riducendo la profondità massima si ottengono gli stessi risultati cambiando gli altri parametri abbiamo peggioramenti
    dt_params = {
        "max_depth": 11,
        "min_samples_split": 20,
        "min_samples_leaf": 7
    }

    local_dt = LocalModel(
        dataset=dataset,
        model_class=DecisionTreeClassifier,
        model_name="Decision Tree"
    )

    local_dt.run(
        save_json="src/hd/data/results/dt/local_dt_model_info.json",
        savefig_learning_curve="src/hd/data/results/dt/learning_curve_local_dt_model.png",
        **dt_params
    )



    ann_params = {
        "module__hidden_size": 63,
        "module__dropout_rate": 0.6359806350639419,
        "max_epochs": 65,
        "lr": 0.0028864634848197596,
        "batch_size": 32
    }
    
    local_ann = LocalModel(
        dataset=ann_dataset,
        model_class=NeuralNetClassifier,
        model_name='ANN'
    )

    local_ann.run(
        save_json='src/hd/data/results/ann/local_ann_model_info.json',
        savefig_learning_curve='src/hd/data/results/ann/learning_curve_local_ann_model.png',
        **ann_params
    )
"""