# Documentazione - HDM

## Struttura del progetto

```bash
.
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    ├── csp
    │   ├── NSP
    │   │   ├── N25
    │   │   ├── pom.xml
    │   │   ├── src
    │   │   │   └── main
    │   │   │       └── java
    │   │   │           └── com
    │   │   │               └── mycompany
    │   │   │                   └── nsp
    │   │   │                     ├── NSP.java
    │   │   │                     ├── NSPGeneticAlgorithm.java
    │   │   │                     ├── NSPGeneticAlgorithmLocalSearch.java
    │   │   │                     ├── NSPTabuSearch.java
    │   │   │                     └── NspData.java
    │   │   └── target
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── data
    │   │   ├── best_params
    │   │   ├── solutions
    │   │   │   ├── genetic_algorithm
    │   │   │   │   ├── ct1
    │   │   │   │   ├── ct2
    │   │   │   │   └── ct3
    │   │   │   ├── genetic_algorithm_local_search
    │   │   │   │   ├── ct1
    │   │   │   │   ├── ct2
    │   │   │   │   └── ct3
    │   │   │   └── tabu
    │   │   └── stats
    │   │       ├── cumulative_execution_times.png
    │   │       ├── fitness_boxplots.png
    │   │       └── time_stats.png
    │   ├── evaluateAlgorithms.py
    │   ├── nsp.py
    │   └── optimizeParams.py
    ├── graph
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── biHospital.py
    │   ├── data
    │   │   ├── biGraph
    │   │   │   ├── all_graph
    │   │   │   │   ├── params
    │   │   │   │   └── stats
    │   │   │   │       ├── agg_search_alg_bigraph.png
    │   │   │   │       ├── cumulative_execution_times_bigraph.png
    │   │   │   │       └── time_stats_bigraph.png
    │   │   │   ├── cmp
    │   │   │   │   ├── astar
    │   │   │   │   │   ├── astar_agg_data.png
    │   │   │   │   │   ├── astar_cet.png
    │   │   │   │   │   └── astar_time_stats.png
    │   │   │   │   ├── bfs
    │   │   │   │   │   ├── bfs_agg_data.png
    │   │   │   │   │   ├── bfs_cet.png
    │   │   │   │   │   └── bfs_time_stats.png
    │   │   │   │   ├── dfbb
    │   │   │   │   │   ├── dfbb_agg_data.png
    │   │   │   │   │   ├── dfbb_cet.png
    │   │   │   │   │   └── dfbb_time_stats.png
    │   │   │   │   ├── dfs
    │   │   │   │   │   ├── dfs_agg_data.png
    │   │   │   │   │   ├── dfs_cet.png
    │   │   │   │   │   └── dfs_time_stats.png
    │   │   │   │   ├── id
    │   │   │   │   │   ├── id_agg_data.png
    │   │   │   │   │   ├── id_cet.png
    │   │   │   │   │   └── id_time_stats.png
    │   │   │   │   └── lcfs
    │   │   │   │       ├── lcfs_agg_data.png
    │   │   │   │       ├── lcfs_cet.png
    │   │   │   │       └── lcfs_time_stats.png
    │   │   │   ├── heuristics.pl
    │   │   │   ├── islands
    │   │   │   │   ├── params
    │   │   │   │   └── stats
    │   │   │   │       ├── isl_agg_search_alg_bigraph.png
    │   │   │   │       ├── isl_cumulative_execution_times_bigraph.png
    │   │   │   │       └── isl_time_stats_bigraph.png
    │   │   │   ├── piano1.png
    │   │   │   ├── piano2.png
    │   │   │   └── piano3.png
    │   │   ├── diGraph
    │   │   │   ├── all_graph
    │   │   │   │   ├── params
    │   │   │   │   └── stats
    │   │   │   │       ├── agg_search_alg_digraph.png
    │   │   │   │       ├── cumulative_execution_times_digraph.png
    │   │   │   │       └── time_stats_digraph.png
    │   │   │   ├── cmp
    │   │   │   │   ├── astar
    │   │   │   │   │   ├── astar_agg_data.png
    │   │   │   │   │   ├── astar_cet.png
    │   │   │   │   │   └── astar_time_stats.png
    │   │   │   │   ├── bfs
    │   │   │   │   │   ├── bfs_agg_data.png
    │   │   │   │   │   ├── bfs_cet.png
    │   │   │   │   │   └── bfs_time_stats.png
    │   │   │   │   ├── dfbb
    │   │   │   │   │   ├── dfbb_agg_data.png
    │   │   │   │   │   ├── dfbb_cet.png
    │   │   │   │   │   └── dfbb_time_stats.png
    │   │   │   │   ├── dfs
    │   │   │   │   │   ├── dfs_agg_data.png
    │   │   │   │   │   ├── dfs_cet.png
    │   │   │   │   │   └── dfs_time_stats.png
    │   │   │   │   ├── id
    │   │   │   │   │   ├── id_agg_data.png
    │   │   │   │   │   ├── id_cet.png
    │   │   │   │   │   └── id_time_stats.png
    │   │   │   │   └── lcfs
    │   │   │   │       ├── lcfs_agg_data.png
    │   │   │   │       ├── lcfs_cet.png
    │   │   │   │       └── lcfs_time_stats.png
    │   │   │   ├── heuristics.pl
    │   │   │   ├── islands
    │   │   │   │   ├── params
    │   │   │   │   └── stats
    │   │   │   │       ├── isl_agg_search_alg_digraph.png
    │   │   │   │       ├── isl_cumulative_execution_times_digraph.png
    │   │   │   │       └── isl_time_stats_digraph.png
    │   │   │   ├── piano1.png
    │   │   │   ├── piano2.png
    │   │   │   └── piano3.png
    │   │   └── hospital.pl
    │   ├── evalUtils.py
    │   ├── evaluateAlgorithms.py
    │   ├── hospital.py
    │   ├── islandSearch.py
    │   └── pathFinder.py
    ├── hd
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── ann.py
    │   ├── compare_models.py
    │   ├── data
    │   │   ├── heart-disease.names
    │   │   ├── notProcessedDataset
    │   │   ├── processedDataset
    │   │   │   └── heart_disease.csv
    │   │   └── results
    │   │       ├── ab
    │   │       │   ├── adaboost_model_info.json
    │   │       │   └── learning_curve_ab.png
    │   │       ├── ann
    │   │       │   ├── ann_model_info.json
    │   │       │   └── learning_curve_ann.png
    │   │       ├── comparison.png
    │   │       ├── dt
    │   │       │   ├── decision_tree_model_info.json
    │   │       │   └── learning_curve_dt.png
    │   │       ├── lr
    │   │       │   ├── learning_curve_lr.png
    │   │       │   └── logistic_regression_model_info.json
    │   │       └── xgb
    │   │           ├── learning_curve_xgb.png
    │   │           └── xgboost_model_info.json
    │   ├── heartDisease.py
    │   └── model.py
    └── util
        ├── __init__.py
        ├── __pycache__
        └── utility.py
```
