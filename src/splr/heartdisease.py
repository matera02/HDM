import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    confusion_matrix, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from urllib.request import urlopen
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Funzione per caricare i dataset da URL
def load_data_from_url(url, column_names):
    data = pd.read_csv(url, header=None, names=column_names)
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.astype(float)
    return data


# URL dei dataset
url_cleveland = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
url_hungarian = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
url_switzerland = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'

# Nomi delle colonne per i dataset
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal', 'num']

# Caricamento dei dataset
cleveland_data = load_data_from_url(url_cleveland, column_names)
#hungarian_data = load_data_from_url(url_hungarian, column_names)
#switzerland_data = load_data_from_url(url_switzerland, column_names)

# Unificazione dei dati
#data = pd.concat([cleveland_data, hungarian_data, switzerland_data])
data = pd.concat([cleveland_data], axis=1)
# Target binario (0: no heart disease, 1: heart disease)
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)


# Divisione dei dati
def split_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_test, y_train, y_test = split_data(data)

# Normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversione in tensori PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# Rete Neurale Artificiale con PyTorch
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 8)
        self.layer2 = nn.Linear(8, 2)
        self.layer3 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


ann_model = ANNModel()

# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.BCELoss()
optimizer = optim.Adam(ann_model.parameters(), lr=0.001)

# Addestramento del modello
num_epochs = 100
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        outputs = ann_model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Funzione per addestrare gli altri modelli
def train_adaboost(X_train, y_train):
    adaboost = AdaBoostClassifier(n_estimators=50, algorithm='SAMME', random_state=42)
    adaboost.fit(X_train, y_train)
    return adaboost


def train_dt(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    return dt


def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model


# Addestramento degli altri modelli
adaboost = train_adaboost(X_train, y_train)
dt = train_dt(X_train, y_train)
xgb_model = train_xgboost(X_train, y_train)


# Funzione per valutare i modelli
def evaluate_model(model, X_test, y_test):
    if isinstance(model, ANNModel):
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_test).numpy().ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return acc, prec, rec, f1, mcc, cm, fpr, tpr, roc_auc


# Valutazione dei modelli
metrics_ann = evaluate_model(ann_model, X_test_tensor, y_test_tensor)
metrics_adaboost = evaluate_model(adaboost, X_test, y_test)
metrics_dt = evaluate_model(dt, X_test, y_test)
metrics_xgb = evaluate_model(xgb_model, X_test, y_test)


#Stampo l'accuracy
print(f'Accuracy ANN: {metrics_ann[0]:.2f}')
print(f'Accuracy AdaBoost: {metrics_adaboost[0]:.2f}')
print(f'Accuracy DT: {metrics_dt[0]:.2f}')
print(f'Accuracy XGB: {metrics_xgb[0]:.2f}')

# Stampa delle precisioni di ciascun modello
print(f'Precision ANN: {metrics_ann[1]:.2f}')
print(f'Precision AdaBoost: {metrics_adaboost[1]:.2f}')
print(f'Precision Decision Tree: {metrics_dt[1]:.2f}')
print(f'Precision XGBoost: {metrics_xgb[1]:.2f}')


# Step 4: Comparazione dei Risultati
# Creazione di tabelle e grafici per la comparazione con i risultati riportati nello studio

# Grafico ROC
plt.figure()
plt.plot(metrics_ann[6], metrics_ann[7], label=f'ANN (AUC = {metrics_ann[8]:.2f})')
plt.plot(metrics_adaboost[6], metrics_adaboost[7], label=f'AdaBoost (AUC = {metrics_adaboost[8]:.2f})')
plt.plot(metrics_dt[6], metrics_dt[7], label=f'DT (AUC = {metrics_dt[8]:.2f})')
plt.plot(metrics_xgb[6], metrics_xgb[7], label=f'XGBoost (AUC = {metrics_xgb[8]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Heart Disease Prediction')
plt.legend()
plt.show()

# Salvataggio dei risultati in un file
results = {
    'ANN': metrics_ann,
    'AdaBoost': metrics_adaboost,
    'DT': metrics_dt,
    'XGBoost': metrics_xgb
}

# Esempio di salvataggio dei risultati
results_df = pd.DataFrame.from_dict(results, orient='index',
                                    columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'Confusion Matrix',
                                             'FPR', 'TPR', 'ROC AUC'])
results_df.to_csv('results.csv')