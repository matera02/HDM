import pickle
import time

import optuna
from concurrent.futures import ThreadPoolExecutor

NTRIALS = 1000
DIRECTION = 'minimize'

TABU = 'TabuSearch'
class Utility:
    @staticmethod
    # best_value si può riferire o alla fitness dei modelli per l'nsp
    # oppure per i modelli di machine learning e deep learning
    def salva_parametri_modello(algorithm, params, fitness, filename='best_params.pkl'):
        try:
            with open(filename, 'rb') as f:
                best_data = pickle.load(f)
        except FileNotFoundError:
            best_data = {}

        best_data[algorithm] = {'params': params, 'fitness': fitness}

        with open(filename, 'wb') as f:
            pickle.dump(best_data, f)

    @staticmethod
    def carica_parametri_modello(algorithm, filename='best_params.pkl'):
        with open(filename, 'rb') as f:
            best_data = pickle.load(f)
        return best_data.get(algorithm)


    @staticmethod
    def get_optimized_params(study_name, optimization_function, direction=DIRECTION,n_trials=NTRIALS, n_jobs=8):
        study = optuna.create_study(study_name=study_name, direction=direction)
        study.optimize(optimization_function, n_trials=n_trials, n_jobs=n_jobs)
        return study.best_params, study.best_value



    @staticmethod
        #n thread per n file su cui scrivere i risultati
    # i file sono numerati da 1 a n
    def write_file_result(input_dir, output_dir,start_file, end_file, max_workers, function):
        files = [f"{input_dir}/{i}.nsp" for i in range(start_file, end_file + 1)] #ultimo estremo escluso
        best_fitness_values = []

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # FAI ATTENZIONE ALLA FUNZIONE IN INPUT
            futures = [executor.submit(function, file, output_dir) for file in files]
            for future in futures:
                best_fitness_values.append(future.result())
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("All file processed")
        print("Best fitness values: ", best_fitness_values)
        print("Elapsed time: ", elapsed_time)


    @staticmethod
        # output_directory è quella dove stanno scritti i risultati per ciascun algoritmo
    def load_results(output_directory, num_files):
        results = []
        for i in range(1, num_files + 1):
            file = f"{output_directory}/{i}.pkl"
            try:
                with open(file, 'rb') as f:
                    result = pickle.load(f)
                    results.append(result)
                    print(f"Loaded {file}: Best Fitness = {result['best_fitness']}, Execution Time = {result['execution_time']} seconds")
                    #print("Hospital coverage:", result['hospital_coverage'])
                    #print("Nurses preferences:", result['nurse_preferences'])
                    #print("Best schedule:", result['best_schedule'])
                    print("\n")
            except FileNotFoundError:
                print(f"File {file} not found.")
        return results
    

    @staticmethod
    def create_pkl_from_nsp(filename, output_directory, result):
        base_filename = filename.split('/')[-1].replace('.nsp', '.pkl')
        output_filename = f"{output_directory}/{base_filename}"
        with open(output_filename, 'wb') as f:
            pickle.dump(result, f)