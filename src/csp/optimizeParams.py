from functools import partial
from nsp import NSP
from src.util.utility import Utility as util

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

def get_params_and_fitness(best_data):
    return best_data['params'], best_data['fitness']


def run_tabu_search(params_tabu):
    iterations = params_tabu['iterations']
    tabu_tenure = params_tabu['tabu_tenure']
    
    print("Iterations: ", iterations)
    print("Tabu tenure: ", tabu_tenure)

    best_schedule, best_fitness = NSP.tabu_search(iterations, tabu_tenure)

    print(best_schedule)
    print()
    print(best_fitness)



def run_genetic_algorithm(params_ga, crossover_type):
    population_size = params_ga['population_size']
    generations = params_ga['generations']
    mutation_rate = params_ga['mutation_rate']

    best_schedule, best_fitness = NSP.genetic_algorithm(population_size, generations, mutation_rate, crossover_type)

    print(best_schedule)
    print()
    print(best_fitness)

def run_genetic_algorithm_local_search(params_gals, crossover_type):
    population_size = params_gals['population_size']
    generations = params_gals['generations']
    mutation_rate = params_gals['mutation_rate']
    local_search_iterations = params_gals['local_search_iterations']

    best_schedule, best_fitness = NSP.genetic_algorithm_local_search(population_size, generations, mutation_rate, 
                                                                     local_search_iterations, crossover_type)
    print(best_schedule)
    print()
    print(best_fitness)


def print_info(description, best_params, best_fitness):
    print(description)
    print("Migliori parametri: ", best_params)
    print("Fitness: ", best_fitness)


def find_best_params():
    filename = "src/csp/data/best_params/best_params.pkl"
    NSP.start()
    
    # OTTIMIZZAZIONE PER LA TABU SEARCH
    best_params, best_fitness = util.get_optimized_params(study_name=TABU, optimization_function=NSP.objective_tabu)
    
    # SALVATAGGIO DEI PARAMETRI PER LA TABU SEARCH
    util.salva_parametri_modello(TABU, best_params, best_fitness, filename)
    
    # OTTIMIZZAZIONE PER L'ALGORITMO GENETICO CON CROSSOVER TYPE 1
    optimization_function = partial(NSP.objective_genetic_algorithm, crossover_type=CROSSOVER_TYPE_1)
    best_params, best_fitness = util.get_optimized_params(study_name=GACT1, optimization_function=optimization_function)

    # SALVATAGGIO DEI PARAMETRI PER L'ALGORITMO GENETICO CON CROSSOVER TYPE 1
    util.salva_parametri_modello(GACT1, best_params, best_fitness, filename)

    # OTTIMIZZAZIONE PER L'ALGORITMO GENETICO CON CROSSOVER TYPE 2
    optimization_function = partial(NSP.objective_genetic_algorithm, crossover_type=CROSSOVER_TYPE_2)
    best_params, best_fitness = util.get_optimized_params(study_name=GACT2, optimization_function=optimization_function)

    # SALVATAGGIO DEI PARAMETRI PER L'ALGORITMO GENETICO CON CROSSOVER TYPE 2
    util.salva_parametri_modello(GACT2, best_params, best_fitness, filename)

    # OTTIMIZZAZIONE PER L'ALGORITMO GENETICO CON CROSSOVER TYPE 3
    optimization_function = partial(NSP.objective_genetic_algorithm, crossover_type=CROSSOVER_TYPE_3)
    best_params, best_fitness = util.get_optimized_params(study_name=GACT3, optimization_function=optimization_function)

    # SALVATAGGIO DEI PARAMETRI PER L'ALGORITMO GENETICO CON CROSSOVER TYPE 3
    util.salva_parametri_modello(GACT3, best_params, best_fitness, filename)

    # OTTIMIZZAZIONE PER L'ALGORITMO GENETICO CON LOCAL SEARCH E CROSSOVER TYPE 1
    optimization_function=partial(NSP.objective_genetic_algorithm_local_search, crossover_type=CROSSOVER_TYPE_1)
    best_params, best_fitness = util.get_optimized_params(study_name=GALSCT1, optimization_function=optimization_function)

    # SALVATAGGIO DEI PARAMETRI PER L'ALGORITMO GENETICO CON LOCAL SEARCH CROSSOVER TYPE 1
    util.salva_parametri_modello(GALSCT1, best_params, best_fitness, filename)

    # OTTIMIZZAZIONE PER L'ALGORITMO GENETICO CON LOCAL SEARCH E CROSSOVER TYPE 2
    optimization_function = partial(NSP.objective_genetic_algorithm_local_search, crossover_type=CROSSOVER_TYPE_2)
    best_params, best_fitness = util.get_optimized_params(study_name=GALSCT2, optimization_function=optimization_function)

    # SALVATAGGIO DEI PARAMETRI PER L'ALGORITMO GENETICO CON LOCAL SEARCH CROSSOVER TYPE 2
    util.salva_parametri_modello(GALSCT2, best_params, best_fitness, filename)

    # OTTIMIZZAZIONE PER L'ALGORITMO GENETICO CON LOCAL SEARCH E CROSSOVER TYPE 3
    optimization_function = partial(NSP.objective_genetic_algorithm_local_search, crossover_type=CROSSOVER_TYPE_3)
    best_params, best_fitness = util.get_optimized_params(study_name=GALSCT3, optimization_function=optimization_function)

    # SALVATAGGIO DEI PARAMETRI PER L'ALGORITMO GENETICO CON LOCAL SEARCH CROSSOVER TYPE 3
    util.salva_parametri_modello(GALSCT3, best_params, best_fitness, filename)

    # CARICO I PARAMETRI DI CIASCUN MODELLO
    best_data_tabu = util.carica_parametri_modello(TABU, filename)
    best_data_gact1 = util.carica_parametri_modello(GACT1, filename)
    best_data_gact2 = util.carica_parametri_modello(GACT2, filename)
    best_data_gact3 = util.carica_parametri_modello(GACT3, filename)
    best_data_galsct1 = util.carica_parametri_modello(GALSCT1, filename)
    best_data_galsct2 = util.carica_parametri_modello(GALSCT2, filename)
    best_data_galsct3 = util.carica_parametri_modello(GALSCT3, filename)

    params_tabu, fitness_tabu = get_params_and_fitness(best_data_tabu)
    params_gact1, fitness_gact1 = get_params_and_fitness(best_data_gact1)
    params_gact2, fitness_gact2 = get_params_and_fitness(best_data_gact2)
    params_gact3, fitness_gact3 = get_params_and_fitness(best_data_gact3)
    params_galsct1, fitness_galsct1 = get_params_and_fitness(best_data_galsct1)
    params_galsct2, fitness_galsct2 = get_params_and_fitness(best_data_galsct2)
    params_galsct3, fitness_galsct3 = get_params_and_fitness(best_data_galsct3)

    print_info("Avvio Tabu Search: ", params_tabu, fitness_tabu)
    run_tabu_search(params_tabu)

    print_info("Avvio Algoritmo genetico con Crossover 1: ", params_gact1, fitness_gact1)
    run_genetic_algorithm(params_gact1, CROSSOVER_TYPE_1)

    print_info("Avvio Algoritmo genetico con Crossover 2: ", params_gact2, fitness_gact2)
    run_genetic_algorithm(params_gact2, CROSSOVER_TYPE_2)

    print_info("Avvio Algoritmo genetico con Crossover 3: ", params_gact3, fitness_gact3)
    run_genetic_algorithm(params_gact3, CROSSOVER_TYPE_3)

    print_info("Avvio Algoritmo genetico con Local Search e Crossover 1: ", params_galsct1, fitness_galsct1)
    run_genetic_algorithm_local_search(params_galsct1, CROSSOVER_TYPE_1)

    print_info("Avvio Algoritmo genetico con Local Search e Crossover 2: ", params_galsct2, fitness_galsct2)
    run_genetic_algorithm_local_search(params_galsct2, CROSSOVER_TYPE_2)

    print_info("Avvio Algoritmo genetico con Local Search e Crossover 3: ", params_galsct3, fitness_galsct3)
    run_genetic_algorithm_local_search(params_galsct3, CROSSOVER_TYPE_3)

    NSP.shutdown()

if __name__ == '__main__':
    find_best_params()




