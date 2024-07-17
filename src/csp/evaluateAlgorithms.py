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
    tabu_search = NSPTabuSearch(filename, iterations, tabu_tenure)
    
    num_nurses, num_days, num_shifts, hospital_coverage, nurse_preferences = get_problem_data(tabu_search)

    start_time = time.time()
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
    pass

def process_file_genetic_algorithm_local_search(filename, output_directory, algorithm, 
                                                crossover_type):
    pass

def create_tabu_files(start_file, end_file):
    util.write_file_result('src/csp/NSP/N25', 'src/csp/data/solutions/tabu',
                           start_file, end_file, 8, process_file_tabu_search)
    util.load_results('src/csp/data/solutions/tabu', end_file) # fino a 100 in questo caso

if __name__ == '__main__':
    NSP.start()
    #process_file_tabu_search("src/csp/NSP/N25/1.nsp", "src/csp/data/solutions/tabu")
    #util.load_results('src/csp/data/solutions/tabu', 1)
    create_tabu_files(1, 1000)
    
    NSP.shutdown()


