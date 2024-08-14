from src.graph.evalUtils import EvaluationUtils as evlutl
from src.graph.hospital import Hospital
from src.graph.biHospital import BiHospital
from src.graph.islandSearch import IDGS
from src.graph.pathFinder import PathFinder
from src.util.utility import Utility as util
from functools import partial

# COSTANTI PER GRAFO DIRETTO
HEURISTIC_DIGRAPH = 'src/graph/data/diGraph/heuristics.pl'

#SAVE_FIG_AGG_DATA = 'src/graph/data/diGraph/agg_search_alg.png'
#SAVE_FIG_CET = 'src/graph/data/diGraph/cumulative_execution_times.png'
#SAVE_FIG_ETS = 'src/graph/data/diGraph/time_stats.png'
#SAVE_FIG_AGG_DATA_ISL = 'src/graph/data/diGraph/agg_search_alg_isl.png'
#SAVE_FIG_CET_ISL = 'src/graph/data/diGraph/cumulative_execution_times_isl.png'
#SAVE_FIG_ETS_ISL = 'src/graph/data/diGraph/time_stats_isl.png'#

# COSTANTI PER GRAFO BIDIREZIONALE
HEURISTIC_BIGRAPH = 'src/graph/data/biGraph/heuristics.pl'
#BI_SAVE_FIG_AGG_DATA = 'src/graph/data/biGraph/bi_agg_search_alg.png'
#BI_SAVE_FIG_CET = 'src/graph/data/biGraph/bi_cumulative_execution_times.png'
#BI_SAVE_FIG_ETS = 'src/graph/data/biGraph/bi_time_stats.png'
#BI_SAVE_FIG_AGG_DATA_ISL = 'src/graph/data/biGraph/bi_agg_search_alg_isl.png'
#BI_SAVE_FIG_CET_ISL = 'src/graph/data/biGraph/bi_cumulative_execution_times_isl.png'
#BI_SAVE_FIG_ETS_ISL = 'src/graph/data/biGraph/bi_time_stats_isl.png'
#BI_SAVE_ALL_DATA = 'src/graph/data/biGraph/bi_all_search_data.pkl'

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

    DI_GRAPH_BFS_FILENAME = 'src/graph/data/diGraph/all_graph/params/digBfs.pkl'
    DI_GRAPH_DFS_FILENAME = 'src/graph/data/diGraph/all_graph/params/digDfs.pkl'
    DI_GRAPH_ID_FILENAME = 'src/graph/data/diGraph/all_graph/params/digId.pkl'
    DI_GRAPH_LCFS_FILENAME = 'src/graph/data/diGraph/all_graph/params/digLcfs.pkl'
    DI_GRAPH_ASTAR_FILENAME = 'src/graph/data/diGraph/all_graph/params/digAstar.pkl'
    DI_GRAPH_DFBB_FILENAME = 'src/graph/data/diGraph/all_graph/params/digDfbb.pkl'
    SAVE_FIG_AGG_DATA = 'src/graph/data/diGraph/all_graph/stats/agg_search_alg_digraph.png'
    SAVE_FIG_CET = 'src/graph/data/diGraph/all_graph/stats/cumulative_execution_times_digraph.png'
    SAVE_FIG_ETS = 'src/graph/data/diGraph/all_graph/stats/time_stats_digraph.png'
    SAVE_ALL_DATA_DI_GRAPH = 'src/graph/data/diGraph/all_graph/params/di_all_search_data.pkl'

    
    # INTERO GRAFO
    evlutl.save_evaluation_params(graph=hospital,algorithm=PathFinder.bfs,filename=DI_GRAPH_BFS_FILENAME)
    evlutl.save_evaluation_params(graph=hospital,algorithm=PathFinder.dfs,filename=DI_GRAPH_DFS_FILENAME)
    evlutl.save_evaluation_params(graph=hospital, algorithm=PathFinder.IterativeDeepening, filename=DI_GRAPH_ID_FILENAME)
    evlutl.save_evaluation_params(graph=hospital, algorithm=PathFinder.lowestCostSearch, filename=DI_GRAPH_LCFS_FILENAME)
    
    aStarSearch = partial(PathFinder.AStarSearch, filename=HEURISTIC_DIGRAPH)
    aStarSearch.__name__ = 'A*'
    evlutl.save_evaluation_params(graph=hospital, algorithm=aStarSearch, filename=DI_GRAPH_ASTAR_FILENAME)
    
    dfbbSearch = partial(PathFinder.DF_branch_and_bound, filename=HEURISTIC_DIGRAPH)
    dfbbSearch.__name__ = 'DFBB'
    evlutl.save_evaluation_params(graph=hospital, algorithm=dfbbSearch, filename=DI_GRAPH_DFBB_FILENAME)

    evlutl.eval(
        bfs_filename=DI_GRAPH_BFS_FILENAME, 
        dfs_filename=DI_GRAPH_DFS_FILENAME, 
        id_filename=DI_GRAPH_ID_FILENAME, 
        lcfs_filename=DI_GRAPH_LCFS_FILENAME, 
        astar_filename=DI_GRAPH_ASTAR_FILENAME, 
        dfbb_filename=DI_GRAPH_DFBB_FILENAME, 
        savefig_agg_data=SAVE_FIG_AGG_DATA, 
        savefig_cet=SAVE_FIG_CET, 
        savefig_ets=SAVE_FIG_ETS, 
        save_all_data=SAVE_ALL_DATA_DI_GRAPH
    )

    # MEDIANTE ISOLE

    ISL_DI_GRAPH_BFS_FILENAME = 'src/graph/data/diGraph/islands/params/isl_digBfs.pkl'
    ISL_DI_GRAPH_DFS_FILENAME = 'src/graph/data/diGraph/islands/params/isl_digDfs.pkl'
    ISL_DI_GRAPH_ID_FILENAME = 'src/graph/data/diGraph/islands/params/isl_digId.pkl'
    ISL_DI_GRAPH_LCFS_FILENAME = 'src/graph/data/diGraph/islands/params/isl_digLcfs.pkl'
    ISL_DI_GRAPH_ASTAR_FILENAME = 'src/graph/data/diGraph/islands/params/isl_digAstar.pkl'
    ISL_DI_GRAPH_DFBB_FILENAME = 'src/graph/data/diGraph/islands/params/isl_digDfbb.pkl'
    ISL_SAVE_FIG_AGG_DATA = 'src/graph/data/diGraph/islands/stats/isl_agg_search_alg_digraph.png'
    ISL_SAVE_FIG_CET = 'src/graph/data/diGraph/islands/stats/isl_cumulative_execution_times_digraph.png'
    ISL_SAVE_FIG_ETS = 'src/graph/data/diGraph/islands/stats/isl_time_stats_digraph.png'
    ISL_SAVE_ALL_DATA_DI_GRAPH = 'src/graph/data/diGraph/islands/params/isl_di_all_search_data.pkl'


    evlutl.save_evaluation_params_idgs(graph=hospital, algorithm='bfs', save=ISL_DI_GRAPH_BFS_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=hospital, algorithm='dfs', save=ISL_DI_GRAPH_DFS_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=hospital, algorithm='id', save=ISL_DI_GRAPH_ID_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=hospital, algorithm='lcfs', save=ISL_DI_GRAPH_LCFS_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=hospital, algorithm='astar', save=ISL_DI_GRAPH_ASTAR_FILENAME, heuristic_filename=HEURISTIC_DIGRAPH)
    evlutl.save_evaluation_params_idgs(graph=hospital, algorithm='dfbb', save=ISL_DI_GRAPH_DFBB_FILENAME, heuristic_filename=HEURISTIC_DIGRAPH)

    evlutl.eval(
        bfs_filename=ISL_DI_GRAPH_BFS_FILENAME, 
        dfs_filename=ISL_DI_GRAPH_DFS_FILENAME, 
        id_filename=ISL_DI_GRAPH_ID_FILENAME, 
        lcfs_filename=ISL_DI_GRAPH_LCFS_FILENAME, 
        astar_filename=ISL_DI_GRAPH_ASTAR_FILENAME, 
        dfbb_filename=ISL_DI_GRAPH_DFBB_FILENAME, 
        savefig_agg_data=ISL_SAVE_FIG_AGG_DATA, 
        savefig_cet=ISL_SAVE_FIG_CET, 
        savefig_ets=ISL_SAVE_FIG_ETS, 
        save_all_data=ISL_SAVE_ALL_DATA_DI_GRAPH
    )


    ##VALUTAZIONE SU GRAFO BIDIREZIONALE
    bi_hospital = BiHospital(di_graph=hospital).get_bi_hospital()

    BI_GRAPH_BFS_FILENAME = 'src/graph/data/biGraph/all_graph/params/bigBfs.pkl'
    BI_GRAPH_DFS_FILENAME = 'src/graph/data/biGraph/all_graph/params/bigDfs.pkl'
    BI_GRAPH_ID_FILENAME = 'src/graph/data/biGraph/all_graph/params/bigId.pkl'
    BI_GRAPH_LCFS_FILENAME = 'src/graph/data/biGraph/all_graph/params/bigLcfs.pkl'
    BI_GRAPH_ASTAR_FILENAME = 'src/graph/data/biGraph/all_graph/params/bigAstar.pkl'
    BI_GRAPH_DFBB_FILENAME = 'src/graph/data/biGraph/all_graph/params/bigDfbb.pkl'
    BI_GRAPH_SAVE_FIG_AGG_DATA = 'src/graph/data/biGraph/all_graph/stats/agg_search_alg_bigraph.png'
    BI_GRAPH_SAVE_FIG_CET = 'src/graph/data/biGraph/all_graph/stats/cumulative_execution_times_bigraph.png'
    BI_GRAPH_SAVE_FIG_ETS = 'src/graph/data/biGraph/all_graph/stats/time_stats_bigraph.png'
    SAVE_ALL_DATA_BI_GRAPH = 'src/graph/data/biGraph/all_graph/params/bi_all_search_data.pkl'

    # INTERO GRAFO
    evlutl.save_evaluation_params(graph=bi_hospital,algorithm=PathFinder.bfs,filename=BI_GRAPH_BFS_FILENAME)
    evlutl.save_evaluation_params(graph=bi_hospital,algorithm=PathFinder.dfs,filename=BI_GRAPH_DFS_FILENAME)
    evlutl.save_evaluation_params(graph=bi_hospital, algorithm=PathFinder.IterativeDeepening, filename=BI_GRAPH_ID_FILENAME)
    evlutl.save_evaluation_params(graph=bi_hospital, algorithm=PathFinder.lowestCostSearch, filename=BI_GRAPH_LCFS_FILENAME)
    
    aStarSearch = partial(PathFinder.AStarSearch, filename=HEURISTIC_BIGRAPH)
    aStarSearch.__name__ = 'A*'
    evlutl.save_evaluation_params(graph=bi_hospital, algorithm=aStarSearch, filename=BI_GRAPH_ASTAR_FILENAME)
    
    dfbbSearch = partial(PathFinder.DF_branch_and_bound, filename=HEURISTIC_BIGRAPH)
    dfbbSearch.__name__ = 'DFBB'
    evlutl.save_evaluation_params(graph=bi_hospital, algorithm=dfbbSearch, filename=BI_GRAPH_DFBB_FILENAME)

    evlutl.eval(
        bfs_filename=BI_GRAPH_BFS_FILENAME, 
        dfs_filename=BI_GRAPH_DFS_FILENAME, 
        id_filename=BI_GRAPH_ID_FILENAME, 
        lcfs_filename=BI_GRAPH_LCFS_FILENAME, 
        astar_filename=BI_GRAPH_ASTAR_FILENAME, 
        dfbb_filename=BI_GRAPH_DFBB_FILENAME, 
        savefig_agg_data=BI_GRAPH_SAVE_FIG_AGG_DATA, 
        savefig_cet=BI_GRAPH_SAVE_FIG_CET, 
        savefig_ets=BI_GRAPH_SAVE_FIG_ETS, 
        save_all_data=SAVE_ALL_DATA_BI_GRAPH
    )

    # MEDIANTE ISOLE

    ISL_BI_GRAPH_BFS_FILENAME = 'src/graph/data/biGraph/islands/params/isl_bigBfs.pkl'
    ISL_BI_GRAPH_DFS_FILENAME = 'src/graph/data/biGraph/islands/params/isl_bigDfs.pkl'
    ISL_BI_GRAPH_ID_FILENAME = 'src/graph/data/biGraph/islands/params/isl_bigId.pkl'
    ISL_BI_GRAPH_LCFS_FILENAME = 'src/graph/data/biGraph/islands/params/isl_bigLcfs.pkl'
    ISL_BI_GRAPH_ASTAR_FILENAME = 'src/graph/data/biGraph/islands/params/isl_bigAstar.pkl'
    ISL_BI_GRAPH_DFBB_FILENAME = 'src/graph/data/biGraph/islands/params/isl_bigDfbb.pkl'
    ISL_BI_GRAPH_SAVE_FIG_AGG_DATA = 'src/graph/data/biGraph/islands/stats/isl_agg_search_alg_bigraph.png'
    ISL_BI_GRAPH_SAVE_FIG_CET = 'src/graph/data/biGraph/islands/stats/isl_cumulative_execution_times_bigraph.png'
    ISL_BI_GRAPH_SAVE_FIG_ETS = 'src/graph/data/biGraph/islands/stats/isl_time_stats_bigraph.png'
    ISL_SAVE_ALL_DATA_BI_GRAPH = 'src/graph/data/biGraph/islands/params/isl_bi_all_search_data.pkl'



    evlutl.save_evaluation_params_idgs(graph=bi_hospital, algorithm='bfs', save=ISL_BI_GRAPH_BFS_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=bi_hospital, algorithm='dfs', save=ISL_BI_GRAPH_DFS_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=bi_hospital, algorithm='id', save=ISL_BI_GRAPH_ID_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=bi_hospital, algorithm='lcfs', save=ISL_BI_GRAPH_LCFS_FILENAME)
    evlutl.save_evaluation_params_idgs(graph=bi_hospital, algorithm='astar', save=ISL_BI_GRAPH_ASTAR_FILENAME, heuristic_filename=HEURISTIC_BIGRAPH)
    evlutl.save_evaluation_params_idgs(graph=bi_hospital, algorithm='dfbb', save=ISL_BI_GRAPH_DFBB_FILENAME, heuristic_filename=HEURISTIC_BIGRAPH)

    evlutl.eval(
        bfs_filename=ISL_BI_GRAPH_BFS_FILENAME, 
        dfs_filename=ISL_BI_GRAPH_DFS_FILENAME, 
        id_filename=ISL_BI_GRAPH_ID_FILENAME, 
        lcfs_filename=ISL_BI_GRAPH_LCFS_FILENAME, 
        astar_filename=ISL_BI_GRAPH_ASTAR_FILENAME, 
        dfbb_filename=ISL_BI_GRAPH_DFBB_FILENAME, 
        savefig_agg_data=ISL_BI_GRAPH_SAVE_FIG_AGG_DATA, 
        savefig_cet=ISL_BI_GRAPH_SAVE_FIG_CET, 
        savefig_ets=ISL_BI_GRAPH_SAVE_FIG_ETS, 
        save_all_data=ISL_SAVE_ALL_DATA_BI_GRAPH
    )
