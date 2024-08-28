import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_model_info(model_name, dir):
    model_name = f"{model_name.replace(' ', '_').lower()}_model_info.json"
    dest = os.path.join(dir, model_name)
    with open(dest, "r") as f:
        return json.load(f)

def round_metric(value, decimals=3):
    return round(value, decimals)

def compare_models(savefig='src/hd/data/results/comparison.png'):

    # Definisci tutti i modelli in una singola lista, inclusi quelli extra
    models = [
        ("Decision Tree", 'src/hd/data/results/dt/'), 
        ("Logistic Regression", 'src/hd/data/results/lr/'), 
        ("Random Forest", 'src/hd/data/results/rf/'), 
        ("Gradient Boosting", 'src/hd/data/results/gb/'), 
        ("ANN", 'src/hd/data/results/ann/'),
        # Aggiungi modelli extra qui
        ("MD8 Decision Tree", 'src/hd/data/results/dt/local_dt_model_info.json', 'myModel'),
        ("ES_ANN", 'src/hd/data/results/ann/early_stopping_ann.json', 'myModel')
    ]

    model_stats = []

    for model in models:
        if len(model) == 3 and model[2] == 'myModel':
            model_info = load_my_model(model[1])
        else:
            model_info = load_model_info(model[0], model[1])

        model_stats.append({
            "Model": model[0],
            "Accuracy": round_metric(model_info["final_accuracy"] if "final_accuracy" in model_info else model_info["accuracy"]),
            "Precision": round_metric(model_info["precision"]),
            "Recall": round_metric(model_info["recall"]),
            "F1 Score": round_metric(model_info["f1_score"])
        })

    df = pd.DataFrame(model_stats)

    # Creo figura e assi
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    # Creo la tabella
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Setto le proprietà della tabella
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    #plt.title("Confronto fra modelli", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()

def load_my_model(filename):
    with open(filename, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    compare_models()
