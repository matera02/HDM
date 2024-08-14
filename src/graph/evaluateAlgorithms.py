from src.graph.evalUtils import EvaluationUtils as evlutl
from src.graph.hospital import Hospital
from src.graph.biHospital import BiHospital
from src.graph.islandSearch import IDGS
from src.graph.pathFinder import PathFinder
from src.util.utility import Utility as util

# COSTANTI PER GRAFO DIRETTO
HEURISTIC_DIGRAPH = 'src/graph/data/diGraph/heuristics.pl'
SAVE_FIG_AGG_DATA = 'src/graph/data/diGraph/agg_search_alg.png'
SAVE_FIG_CET = 'src/graph/data/diGraph/cumulative_execution_times.png'
SAVE_FIG_ETS = 'src/graph/data/diGraph/time_stats.png'
SAVE_FIG_AGG_DATA_ISL = 'src/graph/data/diGraph/agg_search_alg_isl.png'
SAVE_FIG_CET_ISL = 'src/graph/data/diGraph/cumulative_execution_times_isl.png'
SAVE_FIG_ETS_ISL = 'src/graph/data/diGraph/time_stats_isl.png'

# COSTANTI PER GRAFO BIDIREZIONALE
HEURISTIC_BIGRAPH = 'src/graph/data/biGraph/heuristics.pl'
BI_SAVE_FIG_AGG_DATA = 'src/graph/data/biGraph/bi_agg_search_alg.png'
BI_SAVE_FIG_CET = 'src/graph/data/biGraph/bi_cumulative_execution_times.png'
BI_SAVE_FIG_ETS = 'src/graph/data/biGraph/bi_time_stats.png'
BI_SAVE_FIG_AGG_DATA_ISL = 'src/graph/data/biGraph/bi_agg_search_alg_isl.png'
BI_SAVE_FIG_CET_ISL = 'src/graph/data/biGraph/bi_cumulative_execution_times_isl.png'
BI_SAVE_FIG_ETS_ISL = 'src/graph/data/biGraph/bi_time_stats_isl.png'
BI_SAVE_ALL_DATA = 'src/graph/data/biGraph/bi_all_search_data.pkl'

def old_eval():
    hospital = Hospital.get_hospital()
    # VALUTAZIONE ALGORITMI SU GRAFO DIRETTO
    #evlutl.eval(
    #    G=hospital,
    #    heuristic_filename=HEURISTIC_DIGRAPH,
    #    savefig_agg_data=SAVE_FIG_AGG_DATA,
    #    savefig_cet=SAVE_FIG_CET,
    #    savefig_ets=SAVE_FIG_ETS,
    #    save_all_data='src/graph/data/diGraph/all_search_data_digraph.pkl'
    #)

    #evlutl.eval_idgs(
    #    G=hospital,
    #    heuristic_filename=HEURISTIC_DIGRAPH,
    #    savefig_agg_data_isl= SAVE_FIG_AGG_DATA_ISL,
    #    savefig_cet_isl= SAVE_FIG_CET_ISL,
    #    savefig_ets_isl= SAVE_FIG_ETS_ISL,
    #    save_all_data='src/graph/data/diGraph/all_search_data_digraph_idgs.pkl'
    #)

    
    ##VALUTAZIONE SU GRAFO BIDIREZIONALE
    bi_hospital = BiHospital(di_graph=hospital).get_bi_hospital()
    #
#
    #evlutl.eval(
    #    G=bi_hospital,
    #    heuristic_filename=HEURISTIC_BIGRAPH,
    #    savefig_agg_data=BI_SAVE_FIG_AGG_DATA,
    #    savefig_cet=BI_SAVE_FIG_CET,
    #    savefig_ets=BI_SAVE_FIG_ETS,
    #    save_all_data='src/graph/data/biGraph/bi_all_search_data_bigraph.pkl'
    #)
##
    #evlutl.eval_idgs(
    #    G=bi_hospital,
    #    heuristic_filename=HEURISTIC_BIGRAPH,
    #    savefig_agg_data_isl= BI_SAVE_FIG_AGG_DATA_ISL,
    #    savefig_cet_isl= BI_SAVE_FIG_CET_ISL,
    #    savefig_ets_isl= BI_SAVE_FIG_ETS_ISL,
    #    save_all_data='src/graph/data/biGraph/bi_all_search_data_bigraph_idgs.pkl'
    #)


if __name__ == '__main__':
    # VALUTAZIONE ALGORITMI SU GRAFO DIRETTO
    hospital = Hospital.get_hospital()

    # questo lo faccio per ogni algoritmo
    n_paths_bfs, paths_explored_bfs, nodes_visited_bfs, times_bfs, _ = evlutl.get_evaluation_params(hospital, PathFinder.bfs)
    util.save_params_to_pickle(
        n_paths=n_paths_bfs,
        nodes_visited=nodes_visited_bfs,
        paths_explored=paths_explored_bfs,
        times=times_bfs,
        filename='' #DA SOSTITUIRE
    )

    # carico i parametri salvati, costruisco i dizionari che mi servono, plot e infine salvo tutti i parametri ottenuti
    # e cancello quelli vecchi
    idgs = IDGS(hospital)
    n_paths_isl_dfs, paths_explored_isl_dfs, nodes_visited_isl_dfs, times_isl_dfs, _ = evlutl.get_evaluation_params_idgs(idgs, algorithm='dfs')
    util.save_params_to_pickle(
        n_paths=n_paths_isl_dfs,
        nodes_visited=nodes_visited_isl_dfs,
        paths_explored=paths_explored_isl_dfs,
        times=times_isl_dfs,
        filename=''
    )

