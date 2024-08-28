import time
import pickle
from src.util.utility import Utility as util
from functools import partial
from nsp import NSP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


TABU = 'TabuSearch'

GACT1 = 'GeneticAlgorithmCrossoverT1'
GACT2 = 'GeneticAlgorithmCrossoverT2'
GACT3 = 'GeneticAlgorithmCrossoverT3'

GALSCT1 = 'GeneticAlgorithmLocalSearchCrossoverT1'
GALSCT2 = 'GeneticAlgorithmLocalSearchCrossoverT2'
GALSCT3 = 'GeneticAlgorithmLocalSearchCrossoverT3'

CROSSOVER_TYPE_1 = 1
CROSSOVER_TYPE_2 = 2
CROSSOVER_TYPE_3 = 3

BEST_PARAMS_FILE = "src/csp/data/best_params/best_params.pkl"

DIR_TABU_SOLUTIONS = 'src/csp/data/solutions/tabu'
DIR_GACT1_SOLUTIONS = 'src/csp/data/solutions/genetic_algorithm/ct1'
DIR_GACT2_SOLUTIONS = 'src/csp/data/solutions/genetic_algorithm/ct2'
DIR_GACT3_SOLUTIONS = 'src/csp/data/solutions/genetic_algorithm/ct3'
DIR_GALSCT1_SOLUTIONS = 'src/csp/data/solutions/genetic_algorithm_local_search/ct1'
DIR_GALSCT2_SOLUTIONS = 'src/csp/data/solutions/genetic_algorithm_local_search/ct2'
DIR_GALSCT3_SOLUTIONS = 'src/csp/data/solutions/genetic_algorithm_local_search/ct3'

# SAVE FIG CUMULATIVE EXECUTION TIMES
SAVEFIG_CET = 'src/csp/data/stats/cumulative_execution_times.png'
# SAVE FIG EXECUTION TIME STATS
SAVEFIG_ETS = 'src/csp/data/stats/time_stats.png'


def get_problem_data(object):
    num_nurses = np.int64(object.getNumNurses())
    num_days = np.int64(object.getNumDays())
    num_shifts = np.int64(object.getNumShifts())
    hospital_coverage =  NSP.java_array_to_numpy(object.getHospitalCoverage())
    nurse_preferences = NSP.java_array_to_numpy(object.getNursePreferences())
    return num_nurses, num_days, num_shifts, hospital_coverage, nurse_preferences


def process_file_tabu_search(filename, ouput_directory):
    best_data = util.carica_parametri_modello(TABU, filename=BEST_PARAMS_FILE)
    params_from_file = best_data['params']
    iterations = params_from_file['iterations']
    tabu_tenure = params_from_file['tabu_tenure']
    
    from com.mycompany.nsp import NSPTabuSearch
    start_time = time.time()
    tabu_search = NSPTabuSearch(filename, iterations, tabu_tenure)
    
    num_nurses, num_days, num_shifts, hospital_coverage, nurse_preferences = get_problem_data(tabu_search)

    
    tabu_search.run()
    execution_time = time.time() - start_time

    best_schedule = NSP.java_array_to_numpy(tabu_search.getBestSchedule())
    best_fitness = np.double(tabu_search.getBestFitness())

    print(f"Processed {filename} with best fitness {best_fitness} in {execution_time} seconds")

    result = {
        'iterations': iterations,
        'tabu_tenure': tabu_tenure,
        'num_nurses': num_nurses,
        'num_days': num_days,
        'num_shifts': num_shifts,
        'hospital_coverage': hospital_coverage,
        'nurse_preferences': nurse_preferences,
        'best_schedule': best_schedule,
        'best_fitness': best_fitness,
        'execution_time': execution_time
    }

    util.create_pkl_from_nsp(filename, ouput_directory, result)

def process_file_genetic_algorithm(filename, output_directory, algorithm, crossover_type):
    best_data = util.carica_parametri_modello(algorithm, BEST_PARAMS_FILE)
    params_from_file = best_data['params']

    population_size = params_from_file['population_size']
    generations = params_from_file['generations']
    mutation_rate = params_from_file['mutation_rate']

    from com.mycompany.nsp import NSPGeneticAlgorithm
    start_time = time.time()
    ga = NSPGeneticAlgorithm(filename, population_size, generations, mutation_rate,
                                 crossover_type)
    
    num_nurses, num_days, num_shifts, hospital_coverage, nurse_preferences = get_problem_data(ga)
    ga.run()
    execution_time = time.time() - start_time

    best_schedule = NSP.java_array_to_numpy(ga.getBestSchedule())
    best_fitness = np.double(ga.getBestFitness())

    print(f"Processed {filename} with best fitness {best_fitness} in {execution_time} seconds")

    result = {
        'crossover_type': crossover_type,
        'population_size': population_size,
        'generations': generations,
        'mutation_rate': mutation_rate,
        'num_nurses': num_nurses,
        'num_days': num_days,
        'num_shifts': num_shifts,
        'hospital_coverage': hospital_coverage,
        'nurse_preferences': nurse_preferences,
        'best_schedule': best_schedule,
        'best_fitness': best_fitness,
        'execution_time': execution_time
    }

    util.create_pkl_from_nsp(filename, output_directory, result)

def process_file_genetic_algorithm_local_search(filename, output_directory, algorithm, 
                                                crossover_type):
    best_data = util.carica_parametri_modello(algorithm, BEST_PARAMS_FILE)

    params_from_file = best_data['params']
    population_size = params_from_file['population_size']
    generations = params_from_file['generations']
    mutation_rate = params_from_file['mutation_rate']
    local_search_iterations = params_from_file['local_search_iterations']

    from com.mycompany.nsp import NSPGeneticAlgorithmLocalSearch
    start_time = time.time()
    gals = NSPGeneticAlgorithmLocalSearch(filename, population_size, generations, mutation_rate,
                                              local_search_iterations, crossover_type)
    num_nurses, num_days, num_shifts, hospital_coverage, nurse_preferences = get_problem_data(gals)
    gals.run()
    
    execution_time = time.time() - start_time

    best_schedule = NSP.java_array_to_numpy(gals.getBestSchedule())
    best_fitness = np.double(gals.getBestFitness())

    print(f"Processed {filename} with best fitness {best_fitness} in {execution_time} seconds")

    result = {
        'crossover_type': crossover_type,
        'population_size': population_size,
        'generations': generations,
        'mutation_rate': mutation_rate,
        'local_search_iterations': local_search_iterations,
        'num_nurses': num_nurses,
        'num_days': num_days,
        'num_shifts': num_shifts,
        'hospital_coverage': hospital_coverage,
        'nurse_preferences': nurse_preferences,
        'best_schedule': best_schedule,
        'best_fitness': best_fitness,
        'execution_time': execution_time
    }

    util.create_pkl_from_nsp(filename, output_directory, result)


def create_files_to_evaluate(start_file, end_file, source_dir, dest_dir, function, num_workers=8):
    util.write_file_result(source_dir, dest_dir, start_file, end_file, num_workers, function)
    util.load_results(dest_dir, end_file)

def make_solutions_files():
    # Tabu Search
    create_files_to_evaluate(1, 100, 'src/csp/NSP/N25', DIR_TABU_SOLUTIONS, process_file_tabu_search)
 
    source = 'src/csp/NSP/N25'

    # ALGORITMO GENETICO CON CT1    
    dest = DIR_GACT1_SOLUTIONS
    gact1 = partial(process_file_genetic_algorithm, algorithm=GACT1, crossover_type=CROSSOVER_TYPE_1)
    create_files_to_evaluate(1, 100, source, dest, gact1)


    # ALGORITMO GENETICO CON CT2
    dest = DIR_GACT2_SOLUTIONS
    gact2 = partial(process_file_genetic_algorithm, algorithm=GACT2, crossover_type=CROSSOVER_TYPE_2)
    create_files_to_evaluate(1, 100, source, dest, gact2)


    # Algoritmo Genetico CON CT3
    dest = DIR_GACT3_SOLUTIONS
    gact3 = partial(process_file_genetic_algorithm, algorithm=GACT3, crossover_type=CROSSOVER_TYPE_3)
    create_files_to_evaluate(1, 100, source, dest, gact3)

    # Genetico con local search e CT1
    dest = DIR_GALSCT1_SOLUTIONS
    galsct1 = partial(process_file_genetic_algorithm_local_search, algorithm=GALSCT1, crossover_type=CROSSOVER_TYPE_1)
    create_files_to_evaluate(1, 100, source, dest, galsct1)

    # Genetico con local search e CT2
    dest = DIR_GALSCT2_SOLUTIONS
    galsct2 = partial(process_file_genetic_algorithm_local_search, algorithm=GALSCT2, crossover_type=CROSSOVER_TYPE_2)
    create_files_to_evaluate(1, 100, source, dest, galsct2)

    # Genetico con local search e CT2
    dest = DIR_GALSCT3_SOLUTIONS
    galsct3 = partial(process_file_genetic_algorithm_local_search, algorithm=GALSCT3, crossover_type=CROSSOVER_TYPE_3)
    create_files_to_evaluate(1, 100, source, dest, galsct3)

def get_fitness(savefig='src/csp/data/stats/fitness_boxplots.png'):
    key = 'best_fitness'
    title = 'Box Plot dei Valori di Fitness per Algoritmo'
    ylabel = 'Valori di Fitness'
    tabu = util.get_items_from_results(DIR_TABU_SOLUTIONS, key)
    gact1 = util.get_items_from_results(DIR_GACT1_SOLUTIONS, key)
    gact2 = util.get_items_from_results(DIR_GACT2_SOLUTIONS, key)
    gact3 = util.get_items_from_results(DIR_GACT3_SOLUTIONS, key)
    galsct1 = util.get_items_from_results(DIR_GALSCT1_SOLUTIONS, key)
    galsct2 = util.get_items_from_results(DIR_GALSCT2_SOLUTIONS, key)
    galsct3 = util.get_items_from_results(DIR_GALSCT3_SOLUTIONS, key)
    
    # Combino i dati in un unico DataFrame
    data = {
        'Algoritmo': ['TABU']*len(tabu) + ['GACT1']*len(gact1) + ['GACT2']*len(gact2) + ['GACT3']*len(gact3) +
                     ['GALSCT1']*len(galsct1) + ['GALSCT2']*len(galsct2) + ['GALSCT3']*len(galsct3),
        'Valori': tabu + gact1 + gact2 + gact3 + galsct1 + galsct2 + galsct3
    }
    df = pd.DataFrame(data)
    
    # Creo il box plot con seaborn
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Algoritmo', y='Valori', data=df)
    
    # Calcolo e sovrappongo le statistiche
    grouped = df.groupby('Algoritmo')['Valori']
    means = grouped.mean().values
    mins = grouped.min().values
    maxs = grouped.max().values
    medians = grouped.median().values
    q1 = grouped.quantile(0.25).values
    q3 = grouped.quantile(0.75).values
    
    offset = 0.02 * (max(maxs) - min(mins))  # Calcolo un offset basato sul range dei dati
    
    for i, (mean, min_val, max_val, median, q1_val, q3_val) in enumerate(zip(means, mins, maxs, medians, q1, q3)):
        ax.text(i, min_val, f'Min: {min_val:.2f}', horizontalalignment='center', color='blue', weight='semibold', fontsize=8)
        ax.text(i, max_val, f'Max: {max_val:.2f}', horizontalalignment='center', color='green', weight='semibold', fontsize=8)
        ax.text(i, q1_val, f'Q1: {q1_val:.2f}', horizontalalignment='center', color='orange', weight='semibold', fontsize=8)
        ax.text(i, q3_val, f'Q3: {q3_val:.2f}', horizontalalignment='center', color='brown', weight='semibold', fontsize=8)
        
        # Distanziamo leggermente media e mediana
        ax.text(i, mean + offset, f'Media: {mean:.2f}', horizontalalignment='center', color='red', weight='semibold', fontsize=8)
        ax.text(i, median - offset, f'Mediana: {median:.2f}', horizontalalignment='center', color='purple', weight='semibold', fontsize=8)
    
    # Aggiungo le etichette
    plt.title(title, fontsize=14)
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Ruoto le etichette sull'asse x per una migliore leggibilità
    plt.xticks(rotation=45)
    
    # Aggiusto il layout per evitare sovrapposizioni
    plt.tight_layout()
    
    # Salvo il plot come immagine
    plt.savefig(savefig)
    
    # Mostro il plot
    plt.show()

# dizionario algoritmo-tempi di esecuzione
def get_times_dict():
    key = 'execution_time'
    times_dict = {
        'TABU': util.get_items_from_results(DIR_TABU_SOLUTIONS, key),
        'GACT1': util.get_items_from_results(DIR_GACT1_SOLUTIONS, key),
        'GACT2': util.get_items_from_results(DIR_GACT2_SOLUTIONS, key),
        'GACT3': util.get_items_from_results(DIR_GACT3_SOLUTIONS, key),
        'GALSCT1': util.get_items_from_results(DIR_GALSCT1_SOLUTIONS, key),
        'GALSCT2': util.get_items_from_results(DIR_GALSCT2_SOLUTIONS, key),
        'GALSCT3': util.get_items_from_results(DIR_GALSCT3_SOLUTIONS, key)
    }
    return times_dict

def print_model_hyperparameters():
    models = {
        'Tabu Search': TABU,
        'Genetic Algorithm Crossover T1': GACT1,
        'Genetic Algorithm Crossover T2': GACT2,
        'Genetic Algorithm Crossover T3': GACT3,
        'Genetic Algorithm Local Search Crossover T1': GALSCT1,
        'Genetic Algorithm Local Search Crossover T2': GALSCT2,
        'Genetic Algorithm Local Search Crossover T3': GALSCT3
    }

    for model_name, model_key in models.items():
        best_data = util.carica_parametri_modello(model_key, BEST_PARAMS_FILE)
        params = best_data['params']
        
        print(f"Model: {model_name}")
        for param, value in params.items():
            print(f"  {param}: {value}")
        print("\n" + "-"*40 + "\n")


if __name__ == '__main__':
    #NSP.start()
    #process_file_tabu_search("src/csp/NSP/N25/1.nsp", "src/csp/data/solutions/tabu")
    #util.load_results('src/csp/data/solutions/tabu', 1)

    #make_solutions_files()

    # get_results('best_fitness', 'Box Plot dei Valori di Fitness per Algoritmo', 'Valori di Fitness')

    # Box plot di fitness
    #get_fitness()
    #times_dict = get_times_dict()
    #util.plot_cumulative_execution_times(times_dict=times_dict, savefig=SAVEFIG_CET)
    #util.get_execution_time_stats(times_dict=times_dict, savefig=SAVEFIG_ETS)

    #NSP.shutdown()
    print_model_hyperparameters()
    

