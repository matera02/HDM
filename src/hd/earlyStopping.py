from src.hd.model import Model
from src.hd.ann import ANN, ANNDataSet
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from sklearn.model_selection import cross_val_score, learning_curve
import optuna
import json
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import numpy as np
from heartDisease import DataSet

class EarlyStoppingNN(Model):
    def __init__(self, dataset, model_class=NeuralNetClassifier, model_name="ES_ANN", n_trials=100, cv=5, n_jobs=4):
        super().__init__(dataset, model_class, model_name, n_trials, cv, n_jobs)

    def objective(self, trial):
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
            train_split=ValidSplit(0.2),  # Uso il 20% del set di addestramento come validazione
            callbacks=[
                EarlyStopping(
                    monitor='valid_acc',
                    patience=5,
                    lower_is_better=False,
                ),
            ],
        )
        scores = cross_val_score(model, self.dataset.X_train, self.dataset.y_train, cv=self.cv, n_jobs=1)
        return scores.mean()
    

    def run(self, save_json, savefig_learning_curve):
        print(f"Sto eseguendo l'ottimizzazione con Optuna per {self.model_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        best_model = NeuralNetClassifier(
        module=ANN,
        module__input_size=self.dataset.X_train.shape[1],
        module__num_classes=2,
        **study.best_params,
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        train_split=ValidSplit(0.2),
        callbacks=[
            EarlyStopping(
                monitor='valid_acc',
                patience=5,
                lower_is_better=False,),
                ],)
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

        with open(save_json, "w") as f:
            json.dump(model_info, f, indent=4)

if __name__ == '__main__':
    dataset = DataSet()
    ann_dataset = ANNDataSet(dataset)
    es_ann = EarlyStoppingNN(dataset=ann_dataset)
    es_ann.run(
        save_json='src/hd/data/results/ann/early_stopping_ann.json',
        savefig_learning_curve='src/hd/data/results/ann/learning_curve_esnn.png'
    )