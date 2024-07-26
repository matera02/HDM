import json
import pandas as pd
import matplotlib.pyplot as plt

def load_model_info(model_name):
    with open(f"{model_name.replace(' ', '_').lower()}_model_info.json", "r") as f:
        return json.load(f)

def compare_models():
    models = ["Decision Tree", "Logistic Regression", "AdaBoost", "XGBoost", "ANN"]
    model_stats = []
    for model in models:
        model_info = load_model_info(model)
        model_stats.append({
            "Model": model,
            "Best Accuracy": model_info["best_accuracy"],
            "Final Accuracy": model_info["final_accuracy"],
            "Precision": model_info["precision"],
            "Recall": model_info["recall"],
            "F1 Score": model_info["f1_score"]
        })
    
    df = pd.DataFrame(model_stats)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Hide axes
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add a title
    plt.title("Model Comparison", fontsize=16, pad=20)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_models()