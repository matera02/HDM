from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from heartDisease import DataSet
from sklearn.metrics import accuracy_score, classification_report

class Model:
    def __init__(self, dataset: DataSet, model, model_name, params, cv=5, n_jobs=8):
        self.dataset = dataset
        self.model = model
        self.model_name = model_name
        self.params = params
        self.cv = cv
        self.n_jobs = n_jobs
    
    # Funzione per plottare le curve di apprendimento
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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
        return plt
    
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
        analysis.append(f"Train variance: {train_variance:.3f}")
        analysis.append(f"Test variance: {test_variance:.3f}")
        
        # Analisi della stabilità del modello
        if test_std > 0.1:
            analysis.append("Il modello mostra una certa instabilità sui dati di test.")
        else:
            analysis.append("Il modello sembra stabile sui dati di test.")
        return "\n".join(analysis)
    
    def plot_score_distribution(self, train_scores, test_scores):
        plt.figure(figsize=(10, 6))
        plt.title(f"Distribuzione degli score - {self.model_name}")
        plt.boxplot([train_scores[-1], test_scores[-1]], labels=['Train', 'Test'])
        plt.ylabel("Score")
        plt.show()
    


    # Grid Search, training, valutazione e curve di apprendimento per Modello
    def run(self):
        print(f"Running Grid Search for {self.model_name}...")
        model_grid = GridSearchCV(self.model, self.params, cv=self.cv, n_jobs=self.n_jobs)
        model_grid.fit(self.dataset.X_train, self.dataset.y_train)
        print(f"Best parameters for {self.model_name}:", model_grid.best_params_)
        model_best = model_grid.best_estimator_
        model_pred = model_best.predict(self.dataset.X_test)
        print(f"{self.model_name} Accuracy:", accuracy_score(self.dataset.y_test, model_pred))
        print(f"{self.model_name} Classification Report:")
        print(classification_report(self.dataset.y_test, model_pred))
        self.plot_learning_curve(model_best, f"Learning Curve - {self.model_name}", self.dataset.X_train, self.dataset.y_train, ylim=(0.7, 1.01), cv=self.cv, n_jobs=self.n_jobs)
        plt.show()
        # Analisi overfitting per modello
        _, model_train_scores, model_test_scores = learning_curve(model_best, self.dataset.X_train, self.dataset.y_train, cv=self.cv)
        print(f"{self.model_name} Overfitting Analysis: ")
        print(self.analyze_overfitting(model_train_scores, model_test_scores))
    
        # Visualizzazione della distribuzione degli score
        self.plot_score_distribution(model_train_scores, model_test_scores)

if __name__ == '__main__':
    dataset = DataSet()
    
    dt = DecisionTreeClassifier(random_state=42)
    dt_params = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt_model = Model(dataset, dt, "Decision Tree", dt_params)
    dt_model.run()

    lr = LogisticRegression(random_state=42)

    lr_params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    lr_model = Model(dataset, lr, 'Logistic Regression', lr_params)
    lr_model.run()

    ab = AdaBoostClassifier(random_state=42, algorithm='SAMME')

    ab_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }

    ab_model = Model(dataset, ab, 'AdaBoost', ab_params)
    ab_model.run()

    xgbc = xgb.XGBClassifier(random_state=42)

    xgb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    xgb_model = Model(dataset, xgbc, 'XGBoost', xgb_params)
    xgb_model.run()

