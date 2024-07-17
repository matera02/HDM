import jpype
import jpype.imports
import numpy as np
import optuna
import time
from src.util.utility import Utility as util
from functools import partial

FILENAME = "src/csp/NSP/N25/1.nsp"

class NSP:

    # Posso inizializzare la JVM UNA VOLTA SOLA
    @staticmethod
    def start(classpath="src/csp/NSP/target/classes/"):
        jpype.startJVM(classpath=[classpath])

    @staticmethod
    def shutdown():
        jpype.shutdownJVM()

    @staticmethod
    def __java_array_to_numpy(java_array):
        return np.array(java_array)
    

    # Richiamo la tabu search in java
    @staticmethod
    def tabu_search(iterations, tabu_tenure, filename=FILENAME):
        # Avvio la JVM e aggiungo il classpath della directory target/classes
        #jpype.startJVM(classpath=["src/csp/NSP/target/classes/"])
        from com.mycompany.nsp import NSPTabuSearch
        tabu_search = NSPTabuSearch(filename, iterations, tabu_tenure)
        tabu_search.run()
        bestSchedule = NSP.__java_array_to_numpy(tabu_search.getBestSchedule())
        bestFitness = np.double(tabu_search.getBestFitness())
        #jpype.shutdownJVM()
        return bestSchedule, bestFitness
    

    # Richiamo l'algoritmo genetico in java
    @staticmethod
    def genetic_algorithm(population_size, generations, mutation_rate,
                          crossover_type, filename=FILENAME):
        #jpype.startJVM(classpath=["src/csp/NSP/target/classes/"])
        from com.mycompany.nsp import NSPGeneticAlgorithm
        ga = NSPGeneticAlgorithm(filename, population_size, generations, mutation_rate,
                                 crossover_type)
        ga.run()
        bestSchedule = NSP.__java_array_to_numpy(ga.getBestSchedule())

        bestFitness = np.double(ga.getBestFitness())
        #jpype.shutdownJVM()
        return bestSchedule, bestFitness
    

    # Richiamo la versione con la local search dell'algoritmo genetico in java
    @staticmethod
    def genetic_algorithm_local_search(population_size, generations, mutation_rate,
                                       local_search_iterations, crossover_type, filename=FILENAME):
        from com.mycompany.nsp import NSPGeneticAlgorithmLocalSearch
        gals = NSPGeneticAlgorithmLocalSearch(filename, population_size, generations, mutation_rate,
                                              local_search_iterations, crossover_type)
        gals.run()
        bestSchedule = NSP.__java_array_to_numpy(gals.getBestSchedule())
        bestFitness = np.double(gals.getBestFitness())
        return bestSchedule, bestFitness
    

    # DEFINISCO LE FUNZIONI OBIETTIVO PER IL TUNING DEGLI IPERPARAMETRI

    # Per la tabu search
    def objective_tabu(trial):
        iterations = trial.suggest_int('iterations', 50, 200)
        tabu_tenure = trial.suggest_int('tabu_tenure', 3, 10)
        _, best_fitness = NSP.tabu_search(iterations, tabu_tenure)
        return best_fitness

    # Per l'algoritmo genetico
    def objective_genetic_algorithm(trial, crossover_type):
        population_size = trial.suggest_int('population_size', 10, 50)
        generations = trial.suggest_int('generations', 50, 500)
        mutation_rate = trial.suggest_float('mutation_rate', 0.1, 0.9)
        _, best_fitness = NSP.genetic_algorithm(population_size, generations, 
                                                mutation_rate, crossover_type)
        return best_fitness
    
    # Per l'algoritmo genetico con local search
    def objective_genetic_algorithm_local_search(trial, crossover_type):
        population_size = trial.suggest_int('population_size', 10, 50)
        generations = trial.suggest_int('generations', 50, 500)
        mutation_rate = trial.suggest_float('mutation_rate', 0.1, 0.9)
        local_search_iterations = trial.suggest_int('local_search_iterations', 10, 200)
        _, best_fitness = NSP.genetic_algorithm_local_search(population_size, generations, mutation_rate,
                                                             local_search_iterations, crossover_type)
        return best_fitness



def try_algorithms():

    #filename = "../NSP/1.nsp"
    iterations = 100
    tabu_tenure = 3
    #best_schedule, best_fitness = NSP.tabu_search(filename, iterations, tabu_tenure)
    best_schedule, best_fitness = NSP.tabu_search(iterations, tabu_tenure)
    print(best_fitness)
    print(best_schedule)
    population_size = 15
    generations = 200
    mutation_rate = 0.3
    crossover_type = 1
    #best_schedule, best_fitness = NSP.genetic_algorithm(filename, population_size, generations,
    #                                                     mutation_rate,crossover_type)
    best_schedule, best_fitness = NSP.genetic_algorithm(population_size, generations,
                                                         mutation_rate,crossover_type)
    print(best_fitness)
    local_search_iterations = 100

    #best_schedule, best_fitness = NSP.genetic_algorithm_local_search(filename, population_size, generations,
    #                                                                 mutation_rate, local_search_iterations, crossover_type)
    best_schedule, best_fitness = NSP.genetic_algorithm_local_search(population_size, generations,
                                                                     mutation_rate, local_search_iterations, crossover_type)
    print(best_fitness)


def try_study_tabu():
    study = optuna.create_study(study_name="Ottimizzazione degli iperparametri della TabuSearch", direction='minimize')
    study.optimize(NSP.objective_tabu, n_trials=10)
    best_params = study.best_params
    best_fitness = study.best_value
    print("Migliori parametri: ", best_params)
    print("Miglior valore: ", best_fitness)

    util.salva_parametri_modello('TabuSearch', best_params, best_fitness)

    best_data = util.carica_parametri_modello('TabuSearch')

    params_from_file = best_data['params']
    fitness_from_file = best_data['fitness']

    print("Parametri caricati: ", params_from_file)
    print("Fitness con i parametri caricati: ", fitness_from_file)

    iterations = params_from_file['iterations']
    tabu_tenure = params_from_file['tabu_tenure']

    print("Iterations: ", iterations)
    print("Tabu tenure: ", tabu_tenure)

    best_schedule, best_fitness = NSP.tabu_search(iterations, tabu_tenure)

    print(best_schedule)
    print()
    print(best_fitness)


def try_study_genetic_algorithm():
    study = optuna.create_study(study_name="Ottimizzazione degli iperparametri dell'Algoritmo Genetico con Crossover 1", direction='minimize')
    crossover_type = 1
    optimization_function = partial(NSP.objective_genetic_algorithm, crossover_type=crossover_type)
    study.optimize(optimization_function, n_trials=10, n_jobs=2)

    best_params = study.best_params
    best_fitness = study.best_value

    print("Migliori parametri: ", best_params)
    print("Miglior valore: ", best_fitness)

    util.salva_parametri_modello('GeneticAlgorithm', best_params, best_fitness)

    best_data = util.carica_parametri_modello('GeneticAlgorithm')

    params_from_file = best_data['params']
    fitness_from_file = best_data['fitness']

    print("Parametri caricati: ", params_from_file)
    print("Fitness con i parametri caricati: ", fitness_from_file)

    population_size = params_from_file['population_size']
    generations = params_from_file['generations']
    mutation_rate = params_from_file['mutation_rate']

    best_schedule, best_fitness = NSP.genetic_algorithm(population_size, generations, mutation_rate, crossover_type)

    print(best_schedule)
    print()
    print(best_fitness)

def try_study_genetic_local_search():
    study = optuna.create_study(study_name="Ottimizzazione degli iperparametri dell'Algoritmo Genetico con Local Search e con Crossover 1",
                                direction='minimize')
    crossover_type = 1
    optimization_function = partial(NSP.objective_genetic_algorithm_local_search, crossover_type=crossover_type)
    study.optimize(optimization_function, n_trials=10, n_jobs=2)
    
    best_params = study.best_params
    best_fitness = study.best_value

    print("Migliori parametri: ", best_params)
    print("Miglior valore: ", best_fitness)

    util.salva_parametri_modello('GeneticAlgorithmLocalSearch', best_params, best_fitness)

    best_data = util.carica_parametri_modello('GeneticAlgorithmLocalSearch')

    params_from_file = best_data['params']
    fitness_from_file = best_data['fitness']

    print("Parametri caricati: ", params_from_file)
    print("Fitness con i parametri caricati: ", fitness_from_file)

    population_size = params_from_file['population_size']
    generations = params_from_file['generations']
    mutation_rate = params_from_file['mutation_rate']
    local_search_iterations = params_from_file['local_search_iterations']

    best_schedule, best_fitness = NSP.genetic_algorithm_local_search(population_size, generations, mutation_rate, 
                                                                     local_search_iterations, crossover_type)
    print(best_schedule)
    print()
    print(best_fitness)



if __name__ == '__main__':
    start_time = time.time()
    NSP.start()
    try_algorithms()
    
    try_study_tabu()
    try_study_genetic_algorithm()
    try_study_genetic_local_search()

    NSP.shutdown()
    end_time = time.time()

    print("Time elapsed: ", end_time-start_time)







