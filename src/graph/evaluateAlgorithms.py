from src.graph.evalUtils import EvaluationUtils as evlutl
from src.graph.hospital import Hospital
from src.graph.biHospital import BiHospital
from src.graph.pathFinder import PathFinder
from src.util.utility import Utility as util
from functools import partial

# COSTANTI PER GRAFO DIRETTO
HEURISTIC_DIGRAPH = 'src/graph/data/diGraph/heuristics.pl'
## INTERO GRAFO
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
## MEDIANTE ISOLE
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


# COSTANTI PER GRAFO BIDIREZIONALE
HEURISTIC_BIGRAPH = 'src/graph/data/biGraph/heuristics.pl'
## INTERO GRAFO
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
## MEDIANTE ISOLE
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

def evaluate():
    # VALUTAZIONE ALGORITMI SU GRAFO DIRETTO
    hospital = Hospital.get_hospital()

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


    #VALUTAZIONE SU GRAFO BIDIREZIONALE
    bi_hospital = BiHospital(di_graph=hospital).get_bi_hospital()

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

# COSTANTI PER I CONFRONTI DEGLI ALGORITMI
# GRAFO DIRETTO
## BFS
CMP_DI_GRAPH_BFS_AGG_DATA = 'src/graph/data/diGraph/cmp/bfs/bfs_agg_data.png'
CMP_DI_GRAPH_BFS_CET = 'src/graph/data/diGraph/cmp/bfs/bfs_cet.png'
CMP_DI_GRAPH_BFS_TS = 'src/graph/data/diGraph/cmp/bfs/bfs_time_stats.png'
## DFS
CMP_DI_GRAPH_DFS_AGG_DATA = 'src/graph/data/diGraph/cmp/dfs/dfs_agg_data.png'
CMP_DI_GRAPH_DFS_CET = 'src/graph/data/diGraph/cmp/dfs/dfs_cet.png'
CMP_DI_GRAPH_DFS_TS = 'src/graph/data/diGraph/cmp/dfs/dfs_time_stats.png'
## ID
CMP_DI_GRAPH_ID_AGG_DATA = 'src/graph/data/diGraph/cmp/id/id_agg_data.png'
CMP_DI_GRAPH_ID_CET = 'src/graph/data/diGraph/cmp/id/id_cet.png'
CMP_DI_GRAPH_ID_TS = 'src/graph/data/diGraph/cmp/id/id_time_stats.png'
## LCFS
CMP_DI_GRAPH_LCFS_AGG_DATA = 'src/graph/data/diGraph/cmp/lcfs/lcfs_agg_data.png'
CMP_DI_GRAPH_LCFS_CET = 'src/graph/data/diGraph/cmp/lcfs/lcfs_cet.png'
CMP_DI_GRAPH_LCFS_TS = 'src/graph/data/diGraph/cmp/lcfs/lcfs_time_stats.png'
## A*
CMP_DI_GRAPH_ASTAR_AGG_DATA = 'src/graph/data/diGraph/cmp/astar/astar_agg_data.png'
CMP_DI_GRAPH_ASTAR_CET = 'src/graph/data/diGraph/cmp/astar/astar_cet.png'
CMP_DI_GRAPH_ASTAR_TS = 'src/graph/data/diGraph/cmp/astar/astar_time_stats.png'
## DFBB
CMP_DI_GRAPH_DFBB_AGG_DATA = 'src/graph/data/diGraph/cmp/dfbb/dfbb_agg_data.png'
CMP_DI_GRAPH_DFBB_CET = 'src/graph/data/diGraph/cmp/dfbb/dfbb_cet.png'
CMP_DI_GRAPH_DFBB_TS = 'src/graph/data/diGraph/cmp/dfbb/dfbb_time_stats.png'

# GRAFO BIDIREZIONALE
## BFS
CMP_BI_GRAPH_BFS_AGG_DATA = 'src/graph/data/biGraph/cmp/bfs/bfs_agg_data.png'
CMP_BI_GRAPH_BFS_CET = 'src/graph/data/biGraph/cmp/bfs/bfs_cet.png'
CMP_BI_GRAPH_BFS_TS = 'src/graph/data/biGraph/cmp/bfs/bfs_time_stats.png'
## DFS
CMP_BI_GRAPH_DFS_AGG_DATA = 'src/graph/data/biGraph/cmp/dfs/dfs_agg_data.png'
CMP_BI_GRAPH_DFS_CET = 'src/graph/data/biGraph/cmp/dfs/dfs_cet.png'
CMP_BI_GRAPH_DFS_TS = 'src/graph/data/biGraph/cmp/dfs/dfs_time_stats.png'
## ID
CMP_BI_GRAPH_ID_AGG_DATA = 'src/graph/data/biGraph/cmp/id/id_agg_data.png'
CMP_BI_GRAPH_ID_CET = 'src/graph/data/biGraph/cmp/id/id_cet.png'
CMP_BI_GRAPH_ID_TS = 'src/graph/data/biGraph/cmp/id/id_time_stats.png'
## LCFS
CMP_BI_GRAPH_LCFS_AGG_DATA = 'src/graph/data/biGraph/cmp/lcfs/lcfs_agg_data.png'
CMP_BI_GRAPH_LCFS_CET = 'src/graph/data/biGraph/cmp/lcfs/lcfs_cet.png'
CMP_BI_GRAPH_LCFS_TS = 'src/graph/data/biGraph/cmp/lcfs/lcfs_time_stats.png'
## A*
CMP_BI_GRAPH_ASTAR_AGG_DATA = 'src/graph/data/biGraph/cmp/astar/astar_agg_data.png'
CMP_BI_GRAPH_ASTAR_CET = 'src/graph/data/biGraph/cmp/astar/astar_cet.png'
CMP_BI_GRAPH_ASTAR_TS = 'src/graph/data/biGraph/cmp/astar/astar_time_stats.png'
## DFBB
CMP_BI_GRAPH_DFBB_AGG_DATA = 'src/graph/data/biGraph/cmp/dfbb/dfbb_agg_data.png'
CMP_BI_GRAPH_DFBB_CET = 'src/graph/data/biGraph/cmp/dfbb/dfbb_cet.png'
CMP_BI_GRAPH_DFBB_TS = 'src/graph/data/biGraph/cmp/dfbb/dfbb_time_stats.png'


def compare():
    # GRAFO DIRETTO
    
    ## BFS
    evlutl.compare_algorithms(
        alg1_name='BFS', 
        alg1_filename=DI_GRAPH_BFS_FILENAME, 
        alg2_name='ISL-BFS', 
        alg2_filename=ISL_DI_GRAPH_BFS_FILENAME, 
        savefig_agg_data=CMP_DI_GRAPH_BFS_AGG_DATA, 
        savefig_cet=CMP_DI_GRAPH_BFS_CET, 
        savefig_time_stats=CMP_DI_GRAPH_BFS_TS
    )
    
    ## DFS
    evlutl.compare_algorithms(
        alg1_name='DFS', 
        alg1_filename=DI_GRAPH_DFS_FILENAME, 
        alg2_name='ISL-DFS', 
        alg2_filename=ISL_DI_GRAPH_DFS_FILENAME, 
        savefig_agg_data=CMP_DI_GRAPH_DFS_AGG_DATA, 
        savefig_cet=CMP_DI_GRAPH_DFS_CET, 
        savefig_time_stats=CMP_DI_GRAPH_DFS_TS
    )

    ## ID
    evlutl.compare_algorithms(
        alg1_name='ID', 
        alg1_filename=DI_GRAPH_ID_FILENAME, 
        alg2_name='ISL-ID', 
        alg2_filename=ISL_DI_GRAPH_ID_FILENAME, 
        savefig_agg_data=CMP_DI_GRAPH_ID_AGG_DATA, 
        savefig_cet=CMP_DI_GRAPH_ID_CET, 
        savefig_time_stats=CMP_DI_GRAPH_ID_TS
    )

    ## LCFS
    evlutl.compare_algorithms(
        alg1_name='LCFS',
        alg1_filename=DI_GRAPH_LCFS_FILENAME,
        alg2_name='ISL-LCFS', 
        alg2_filename=ISL_DI_GRAPH_LCFS_FILENAME, 
        savefig_agg_data=CMP_DI_GRAPH_LCFS_AGG_DATA, 
        savefig_cet=CMP_DI_GRAPH_LCFS_CET, 
        savefig_time_stats=CMP_DI_GRAPH_LCFS_TS
    )

    ## A*
    evlutl.compare_algorithms(
        alg1_name='A*', 
        alg1_filename=DI_GRAPH_ASTAR_FILENAME, 
        alg2_name='ISL-A*', 
        alg2_filename=ISL_DI_GRAPH_ASTAR_FILENAME, 
        savefig_agg_data=CMP_DI_GRAPH_ASTAR_AGG_DATA, 
        savefig_cet=CMP_DI_GRAPH_ASTAR_CET, 
        savefig_time_stats=CMP_DI_GRAPH_ASTAR_TS
    )

    ## DFBB
    evlutl.compare_algorithms(
        alg1_name='DFBB', 
        alg1_filename=DI_GRAPH_DFBB_FILENAME, 
        alg2_name='ISL-DFBB', 
        alg2_filename=ISL_DI_GRAPH_DFBB_FILENAME, 
        savefig_agg_data=CMP_DI_GRAPH_DFBB_AGG_DATA, 
        savefig_cet=CMP_DI_GRAPH_DFBB_CET, 
        savefig_time_stats=CMP_DI_GRAPH_DFBB_TS
    )

    # GRAFO BIDIREZIONALE
    ## BFS
    evlutl.compare_algorithms(
        alg1_name='BFS', 
        alg1_filename=BI_GRAPH_BFS_FILENAME, 
        alg2_name='ISL-BFS', 
        alg2_filename=ISL_BI_GRAPH_BFS_FILENAME, 
        savefig_agg_data=CMP_BI_GRAPH_BFS_AGG_DATA, 
        savefig_cet=CMP_BI_GRAPH_BFS_CET, 
        savefig_time_stats=CMP_BI_GRAPH_BFS_TS
    )
    
    ## DFS
    evlutl.compare_algorithms(
        alg1_name='DFS', 
        alg1_filename=BI_GRAPH_DFS_FILENAME, 
        alg2_name='ISL-DFS', 
        alg2_filename=ISL_BI_GRAPH_DFS_FILENAME, 
        savefig_agg_data=CMP_BI_GRAPH_DFS_AGG_DATA, 
        savefig_cet=CMP_BI_GRAPH_DFS_CET, 
        savefig_time_stats=CMP_BI_GRAPH_DFS_TS
    )

    ## ID
    evlutl.compare_algorithms(
        alg1_name='ID', 
        alg1_filename=BI_GRAPH_ID_FILENAME, 
        alg2_name='ISL-ID', 
        alg2_filename=ISL_BI_GRAPH_ID_FILENAME, 
        savefig_agg_data=CMP_BI_GRAPH_ID_AGG_DATA, 
        savefig_cet=CMP_BI_GRAPH_ID_CET, 
        savefig_time_stats=CMP_BI_GRAPH_ID_TS
    )

    ## LCFS
    evlutl.compare_algorithms(
        alg1_name='LCFS',
        alg1_filename=BI_GRAPH_LCFS_FILENAME,
        alg2_name='ISL-LCFS', 
        alg2_filename=ISL_BI_GRAPH_LCFS_FILENAME, 
        savefig_agg_data=CMP_BI_GRAPH_LCFS_AGG_DATA, 
        savefig_cet=CMP_BI_GRAPH_LCFS_CET, 
        savefig_time_stats=CMP_BI_GRAPH_LCFS_TS
    )

    ## A*
    evlutl.compare_algorithms(
        alg1_name='A*', 
        alg1_filename=BI_GRAPH_ASTAR_FILENAME, 
        alg2_name='ISL-A*', 
        alg2_filename=ISL_BI_GRAPH_ASTAR_FILENAME, 
        savefig_agg_data=CMP_BI_GRAPH_ASTAR_AGG_DATA, 
        savefig_cet=CMP_BI_GRAPH_ASTAR_CET, 
        savefig_time_stats=CMP_BI_GRAPH_ASTAR_TS
    )

    ## DFBB
    evlutl.compare_algorithms(
        alg1_name='DFBB', 
        alg1_filename=BI_GRAPH_DFBB_FILENAME, 
        alg2_name='ISL-DFBB', 
        alg2_filename=ISL_BI_GRAPH_DFBB_FILENAME, 
        savefig_agg_data=CMP_BI_GRAPH_DFBB_AGG_DATA, 
        savefig_cet=CMP_BI_GRAPH_DFBB_CET, 
        savefig_time_stats=CMP_BI_GRAPH_DFBB_TS
    )


if __name__ == '__main__':
    #evaluate()
    #compare()
    pass