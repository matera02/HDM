import pandas as pd
if __name__ == '__main__':
    # Caricare i file
    cleveland_path = 'src/splr/data/notProcessedDataset/cleveland.data'
    hungarian_path = 'src/splr/data/notProcessedDataset/hungarian.data'
    switzerland_path = 'src/splr/data/notProcessedDataset/switzerland.data'
    va_path = 'src/splr/data/notProcessedDataset/va.data'
    
    # Definire i nomi delle colonne (basato su standard disponibili)
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

    # Caricare i dati nei DataFrame
    cleveland_df = pd.read_csv(cleveland_path, names=column_names, na_values='?')
    hungarian_df = pd.read_csv(hungarian_path, names=column_names, na_values='?')
    switzerland_df = pd.read_csv(switzerland_path, names=column_names, na_values='?')
    va_df = pd.read_csv(va_path, names=column_names, na_values='?')

    # Unire i DataFrame
    combined_df = pd.concat([cleveland_df, hungarian_df, switzerland_df, va_df], ignore_index=True)

    # Sostituire i valori mancanti con la mediana della colonna
    combined_df.fillna(combined_df.mean(), inplace=True)
    
    # Verificare la presenza di valori mancanti
    print(combined_df.isnull().sum())

    print(combined_df)

    # Salvare il DataFrame in un file CSV
    output_path = 'src/splr/data/processedDataset/heart_disease.csv'
    combined_df.to_csv(output_path, index=False)






