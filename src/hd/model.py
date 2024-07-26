from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from heartDisease import DataSet
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import json
import os

class Model:
    def __init__(self, dataset: DataSet, model, model_name, params, cv=5, n_jobs=8):
        self.dataset = dataset
        self.model = model
        self.model_name = model_name
        self.params = params
        self.cv = cv
        self.n_jobs = n_jobs
    
    # Metodo per plottare le curve di apprendimento
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), savefig = ''):
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
    
    # Metodo per l'analisi dell'overfitting
    def analyze_overfitting(self, train_scores, test_scores):
        train_mean = np.mean(train_scores[-1])
        test_mean = np.mean(test_scores[-1])
        train_std = np.std(train_scores[-1])
        test_std = np.std(test_scores[-1])
        gap = train_mean - test_mean
        
        analysis = []
        if gap > 0.1:
            analysis.append(f"Possibile overfitting. Gap tra train e test: {gap:.3f}")
        elif train_mean < 0.8:
            analysis.append(f"Possibile underfitting. Train score: {train_mean:.3f}")
        else:
            analysis.append("Il modello sembra ben bilanciato.")
    
        analysis.append(f"Train score: media = {train_mean:.3f}, std = {train_std:.3f}")
        analysis.append(f"Test score: media = {test_mean:.3f}, std = {test_std:.3f}")
    
        # Calcolo della varianza
        train_variance = np.var(train_scores[-1])
        test_variance = np.var(test_scores[-1])
        analysis.append(f"Train varianza: {train_variance:.3f}")
        analysis.append(f"Test varianza: {test_variance:.3f}")
        
        # Analisi della stabilità del modello
        if test_std > 0.1:
            analysis.append("Il modello mostra una certa instabilità sui dati di test.")
        else:
            analysis.append("Il modello sembra stabile sui dati di test.")
        return "\n".join(analysis)
    
    def plot_score_distribution(self, train_scores, test_scores, savefig):
        plt.figure(figsize=(10, 6))
        plt.title(f"Distribuzione degli score - {self.model_name}")
        plt.boxplot([train_scores[-1], test_scores[-1]], labels=['Train', 'Test'])
        plt.ylabel("Score")
        plt.savefig(savefig)
        plt.show()
    
    
    # Grid Search, training, valutazione e curve di apprendimento per Modello
    def run(self, dir_json, savefig_learning_curve, savefig_score_distribution):
        print(f"Running Grid Search for {self.model_name}...")
        model_grid = GridSearchCV(self.model, self.params, cv=self.cv, n_jobs=self.n_jobs)
        model_grid.fit(self.dataset.X_train, self.dataset.y_train)
        print(f"Best parameters for {self.model_name}:", model_grid.best_params_)
        model_best = model_grid.best_estimator_
        model_pred = model_best.predict(self.dataset.X_test)
        
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
        
        print(f"Plotting learning curve for {self.model_name}...")
        train_sizes, train_scores, test_scores = learning_curve(
            model_best, self.dataset.X_train, self.dataset.y_train, 
            cv=self.cv, n_jobs=self.n_jobs, train_sizes=np.linspace(.1, 1.0, 5)
        )
        self.plot_learning_curve(model_best, f"Learning Curve - {self.model_name}", 
                                 self.dataset.X_train, self.dataset.y_train, 
                                 ylim=(0.7, 1.01), cv=self.cv, n_jobs=self.n_jobs, savefig=savefig_learning_curve)
        
        print(f"{self.model_name} Overfitting Analysis: ")
        overfitting_analysis = self.analyze_overfitting(train_scores, test_scores)
        print(overfitting_analysis)
    
        # Visualizzazione della distribuzione degli score
        self.plot_score_distribution(train_scores, test_scores, savefig=savefig_score_distribution)

        # Salvataggio delle informazioni del modello
        model_info = {
            "model_name": self.model_name,
            "best_params": model_grid.best_params_,
            "best_accuracy": model_grid.best_score_,
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

if __name__ == '__main__':
    dataset = DataSet()
    
    dt = DecisionTreeClassifier(random_state=42)
    dt_params = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt_model = Model(dataset, dt, "Decision Tree", dt_params)
    dt_model.run(dir_json='src/hd/data/results/dt/', savefig_learning_curve='src/hd/data/results/dt/learning_curve_dt.png', 
                 savefig_score_distribution='src/hd/data/results/dt/score_distribution_dt.png')

    lr = LogisticRegression(random_state=42)

    lr_params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    lr_model = Model(dataset, lr, 'Logistic Regression', lr_params)
    lr_model.run(dir_json='src/hd/data/results/lr/', savefig_learning_curve='src/hd/data/results/lr/learning_curve_lr.png', 
                 savefig_score_distribution='src/hd/data/results/lr/score_distribution_lr.png')

    ab = AdaBoostClassifier(random_state=42, algorithm='SAMME')

    ab_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }

    ab_model = Model(dataset, ab, 'AdaBoost', ab_params)
    ab_model.run(dir_json='src/hd/data/results/ab/', savefig_learning_curve='src/hd/data/results/ab/learning_curve_ab.png', 
                 savefig_score_distribution='src/hd/data/results/ab/score_distribution_ab.png')

    xgbc = xgb.XGBClassifier(random_state=42)

    xgb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    xgb_model = Model(dataset, xgbc, 'XGBoost', xgb_params)
    xgb_model.run(dir_json='src/hd/data/results/xgb/', savefig_learning_curve='src/hd/data/results/xgb/learning_curve_xgb.png',
                  savefig_score_distribution='src/hd/data/results/xgb/score_distribution_xgb.png')

