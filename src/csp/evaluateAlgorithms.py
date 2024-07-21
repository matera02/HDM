import time
import pickle
from src.util.utility import Utility as util
from functools import partial
from nsp import NSP
import numpy as np

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
    pass


def create_files_to_evaluate(start_file, end_file, source_dir, dest_dir, function, num_workers=8):
    util.write_file_result(source_dir, dest_dir, start_file, end_file, num_workers, function)
    util.load_results(dest_dir, end_file)

def make_solutions_files():
    # Tabu Search
    create_files_to_evaluate(1, 100, 'src/csp/NSP/N25', 'src/csp/data/solutions/tabu', process_file_tabu_search)
 
    source = 'src/csp/NSP/N25'

    # ALGORITMO GENETICO CON CT1    
    dest = 'src/csp/data/solutions/genetic_algorithm/ct1'
    gact1 = partial(process_file_genetic_algorithm, algorithm=GACT1, crossover_type=CROSSOVER_TYPE_1)
    create_files_to_evaluate(1, 100, source, dest, gact1)


    # ALGORITMO GENETICO CON CT2
    dest = 'src/csp/data/solutions/genetic_algorithm/ct2'
    gact2 = partial(process_file_genetic_algorithm, algorithm=GACT2, crossover_type=CROSSOVER_TYPE_2)
    create_files_to_evaluate(1, 100, source, dest, gact2)


    # Algoritmo Genetico CON CT3
    dest = 'src/csp/data/solutions/genetic_algorithm/ct3'
    gact3 = partial(process_file_genetic_algorithm, algorithm=GACT3, crossover_type=CROSSOVER_TYPE_3)
    create_files_to_evaluate(1, 100, source, dest, gact3)

    # Genetico con local search e CT1
    dest = 'src/csp/data/solutions/genetic_algorithm_local_search/ct1'
    galsct1 = partial(process_file_genetic_algorithm_local_search, algorithm=GALSCT1, crossover_type=CROSSOVER_TYPE_1)
    create_files_to_evaluate(1, 100, source, dest, galsct1)

    # Genetico con local search e CT2
    dest = 'src/csp/data/solutions/genetic_algorithm_local_search/ct2'
    galsct2 = partial(process_file_genetic_algorithm_local_search, algorithm=GALSCT2, crossover_type=CROSSOVER_TYPE_2)
    create_files_to_evaluate(1, 100, source, dest, galsct2)

    # Genetico con local search e CT2
    dest = 'src/csp/data/solutions/genetic_algorithm_local_search/ct3'
    galsct3 = partial(process_file_genetic_algorithm_local_search, algorithm=GALSCT3, crossover_type=CROSSOVER_TYPE_3)
    create_files_to_evaluate(1, 100, source, dest, galsct3)


if __name__ == '__main__':
    NSP.start()
    #process_file_tabu_search("src/csp/NSP/N25/1.nsp", "src/csp/data/solutions/tabu")
    #util.load_results('src/csp/data/solutions/tabu', 1)

    #make_solutions_files()
    dest = 'src/csp/data/solutions/genetic_algorithm/ct1'
    util.load_results(dest, 100)

    NSP.shutdown()


