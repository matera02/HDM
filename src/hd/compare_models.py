import json
import pandas as pd
import matplotlib.pyplot as plt
import os
def load_model_info(model_name, dir):
    model_name = f"{model_name.replace(' ', '_').lower()}_model_info.json"
    dest = os.path.join(dir, model_name)
    with open(dest, "r") as f:
        return json.load(f)

def compare_models(savefig='src/hd/data/results/comparison.png'):
    models = [
        ("Decision Tree", 'src/hd/data/results/dt/'), 
        ("Logistic Regression", 'src/hd/data/results/lr/'), 
        ("AdaBoost", 'src/hd/data/results/ab/'), 
        ("XGBoost", 'src/hd/data/results/xgb/'), 
        ("ANN", 'src/hd/data/results/ann/')
    ]
    model_stats = []
    for model in models:
        model_info = load_model_info(model[0], model[1])
        model_stats.append({
            "Model": model[0],
            "Best Accuracy": model_info["best_accuracy"],
            "Final Accuracy": model_info["final_accuracy"],
            "Precision": model_info["precision"],
            "Recall": model_info["recall"],
            "F1 Score": model_info["f1_score"]
        })
    
    df = pd.DataFrame(model_stats)
    
    # Creo figura e assi
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Nascondo asse
    ax.axis('off')
    
    # Creo la tabella
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Setto le proprietà della tabella
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Titolo
    plt.title("Confronto fra modelli", fontsize=16, pad=20)
    
    # Mostro il plot
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()

if __name__ == "__main__":
    compare_models()