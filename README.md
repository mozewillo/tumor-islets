# tumor_islets

Package for IF data graph-based analysis

to install download package and run `pip install -e . `

('.' is path to directory where setup.py is )

### Try running example on your sample
```
import tumor_islets.graph.Graph as graph
import tumor_islets.plots.sample_all as plots
import pandas as pd

data = pd.read_csv(tsv_file_path, sep='\t')

G = graph.Graph(filename='test', panel='1', initial_structure=data)

G.calculate_number_of_neighbors()
plots.visualise_num_column(G.initial_structure, 'n.neighbors', 
                                sample_name='Test', 
                                title='Number of neighbors', 
                                ax=None, panel='IF1')

# calculate marker homogeneity score
G.calculate_marker_homogeneity_score(positive_marker='CD3')
plots.visualise_num_column(G.initial_structure, 'CD3.homogenic.connections.score', sample_name='Test', 
                     title='CD3 connection homogeneity', ax=None, panel='IF1')

# calculate connections score for two markers
G.calculate_marker_connection_score(marker1='CK', marker2='CD3')
plots.visualise_num_column(G.initial_structure, 'CK_to_CD3', sample_name='Test', 
                     title='CK to CD3 connections', ax=None, panel='IF1')
                     
# calculate components based on markers
G.connected_components(positive_marker=['CD3', 'CD20'], add_to_summary=True, max_dist=30, marker_rule='or')
plots.visualise_components(G.initial_structure, sample_name='Test', 
                            column_name='CD20.homogenic.connections.score.component_number', 
                            size_thr_down=3, size_thr_up=None)

# calculate components based on numeric column
G.calculate_marker_homogeneity_score(positive_marker='CD20')
G.connected_components(numeric_column='CD20.homogenic.connections.score', numeric_threshold=0.5, add_to_summary=False, max_dist=20 )
plots.visualise_components(G.initial_structure, sample_name='Test', 
                        column_name='CD20.homogenic.connections.score.component_number',
                         size_thr_down=3, size_thr_up=None)
                 
```


