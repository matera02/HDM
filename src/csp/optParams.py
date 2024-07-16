from functools import partial
TABU = 'TabuSearch'

GACT1 = 'GeneticAlgorithmCrossoverT1'
GACT2 = 'GeneticAlgorithmCrossoverT2'
GACT3 = 'GeneticAlgorithmCrossoverT3'

GALSCT1 = 'GeneticAlgorithmLocalSearchCrossoverT1'
GALSCT2 = 'GeneticAlgorithmLocalSearchCrossoverT2'
GALSCT3 = 'GeneticAlgorithmLocalSearchCrossoverT3'

DIRECTION = 'minimize'

TRIALS = 100

CROSSOVER_TYPE_1 = 1
CROSSOVER_TYPE_2 = 2
CROSSOVER_TYPE_3 = 3


FILENAME = "../NSP/1.nsp"

def get_params_and_fitness(best_data):
    return best_data['params'], best_data['fitness']


def run_tabu(params_tabu):
    pass

def run_ga(params_ga, crossover_type):
    pass

def run_gals(params_gals, crossover_type):
    pass

def print_info(description, best_params, best_fitness):
    print(description)
    print("Migliori parametri: ", best_params)
    print("Fitness: ", best_fitness)



