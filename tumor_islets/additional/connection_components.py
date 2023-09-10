from sklearn.neighbors import radius_neighbors_graph
import pandas as pd
import numpy as np
import multiprocessing


def count_neighbors_df(df, celltype1, celltype2, max_dist):
    points_list = list(zip(df["nucleus.x"], df["nucleus.y"]))
    if 'new.index' in df:
        df.drop(columns=['new.index'], inplace=True, axis=1)
    df.reset_index(inplace=True, drop=True)
    df.index.name = 'new.index'
    df.reset_index(inplace=True)
    i0 = df.loc[df['celltype'] == celltype1]['new.index']
    i1 = df.loc[df['celltype'] == celltype2]['new.index']
    G_temp = radius_neighbors_graph(points_list,
                                    max_dist,
                                    mode="connectivity",
                                    include_self=True,
                                    n_jobs=int(0.75 * multiprocessing.cpu_count()))
    G_n = G_temp[i0, :]
    G_temp = G_temp[i0, :][:, i1]
    counts = G_temp.sum(axis=1)
    if celltype1 == celltype2 or celltype1 == 'all' or celltype2 == 'all':
        counts = counts - 1  # (-1 because each cell is neighboring with itself)
    all_counts = G_n.sum(axis=1)  # calculate number of all neighbors of celltype1
    all_counts = all_counts - 1  # (-1 because each cell is neighboring with itself)
    return counts, all_counts


def flatten_matrix(df, sample_id, columns_ordered, index_column='cell_type', take_nan=False):
    """Flatten matrix to vector, order by columns ordered to keep consistency with other samples"""
    possible_pairs = []
    i = 0
    dif = [i for i in columns_ordered if i not in df.columns]  # check if all values were calulated
    if len(dif) > 0:
        print(f'{sample_id} : Missing columns: {dif}')
        if take_nan:
            for d in dif:
                df = df.T
                df[d] = np.nan
                df = df.T
                df[d] = np.nan
        else:
            return None
    for c1 in columns_ordered:
        for c2 in columns_ordered:
            possible_pairs.append((c1, c2))
        if index_column not in df.columns and df.index.name == index_column:
            df.reset_index(inplace=True)
        ind = [index_column]
        ind.extend(columns_ordered)
        df = df[ind]
        for i, ct in enumerate(columns_ordered):
            v = df.loc[df[index_column] == ct].iloc[0][1:].to_numpy()
            if i == 0:
                sample_vec = v
            else:
                sample_vec = np.hstack([sample_vec, v])

    return sample_vec


def calculate_all_pairs_neighbor_counts_df(df, save=False, sample_name=None, max_dist=None):
    cts = df["celltype"].unique()
    neighbor_matrix = np.zeros((len(cts), len(cts)))
    freq_matrix = np.zeros((len(cts), len(cts)))
    mean_all_matrix = np.zeros((len(cts), len(cts)))

    for i, ct1 in enumerate(cts):
        for j, ct2 in enumerate(cts):
            c, ac = count_neighbors_df(df, ct1, ct2, max_dist=max_dist)
            if c is not None and ac is not None:
                mean_all_matrix[i][j] = c.sum() / ac.sum()
                ct1_n = len(df.loc[df['celltype'] == ct1])
                neighbor_matrix[i][j] = c[c.nonzero()].mean()
                freq_matrix[i][j] = len(c.nonzero()[0]) / ct1_n
            else:
                print("Count neighbors returns None, check parameters")
                mean_all_matrix[i][j] = np.nan
                neighbor_matrix[i][j] = np.nan
                freq_matrix[i][j] = np.nan

    neighbor_matrix = pd.DataFrame(neighbor_matrix, columns=cts, index=cts)
    freq_matrix = pd.DataFrame(freq_matrix, columns=cts, index=cts)
    mean_all_matrix = pd.DataFrame(mean_all_matrix, columns=cts, index=cts)
    neighbor_matrix.index.name = 'cell_type'
    freq_matrix.index.name = 'cell_type'
    mean_all_matrix.index.name = 'cell_type'

    return neighbor_matrix, freq_matrix, mean_all_matrix


def summarize_connections_per_component(self, component_column, mapping_file=None,
                                        max_dist=None, directory='./', include_neighbors=True,
                                        min_cells=20, columns_ordered=None):
    """"""
    if 'celltype' not in self.initial_structure.columns:
        if mapping_file is not None:
            self.map_phenotype_to_celltype(mapping=mapping_file)
    all_samples = None
    celltypes = list(self.initial_structure['celltype'].unique())
    if columns_ordered is None:
        columns_ordered = celltypes
    possible_pairs = []
    samples_list = []

    for c1 in celltypes:
        for c2 in celltypes:
            possible_pairs.append((c1, c2))
    i = 0
    for cm in self.initial_structure[component_column].unique():
        if cm == -1:
            continue

        cdata = self.initial_structure.loc[self.initial_structure[component_column] == cm]
        if len(cdata) < min_cells:
            continue

        if include_neighbors:
            cdata = get_component_with_neighbors(self, column_name=component_column,
                                                 component_number=cm, max_dist=max_dist)
        nm, fm, ma = calculate_all_pairs_neighbor_counts_df(cdata, sample_name=cm, max_dist=max_dist)
        samples_list.append(cm)
        print(fm)
        print(fm.columns)
        vec = flatten_matrix(fm, sample_id=f'Component {cm}',
                             columns_ordered=columns_ordered, index_column='cell_type', take_nan=True)
        if i == 0:
            all_samples = vec.reshape(1,-1)
        else:
            all_samples = np.vstack([all_samples, vec])
        i += 1
    if all_samples is None:
        return None
    all_samples = all_samples.astype('float')
    possible_pairs_str = [f'{p[0]}_{p[1]}' for p in possible_pairs]
#     return all_samples, possible_pairs_str, samples_list
    all_samples_df = pd.DataFrame(all_samples, index=samples_list, columns=possible_pairs_str)
    all_samples_df.index.name = 'component_number'

    return all_samples_df


def get_cells_neighbors(self, cells_selected=None, selected_inds=None,
                        max_dist=None, index_col='index', mode='connectivity'):
    """for given cells (rows from df) or selected indices find all neighbors in given distance
        Returns: neighboring cells indices and selected neighborhood graph"""
    if cells_selected is not None and selected_inds is None:
        selected_inds = list(cells_selected['index'])
    elif selected_inds is None and cells_selected is None:
        print('No cells selected')
        return None
    if max_dist is None or max_dist == self.max_dist:
        D = self._graph_dict
        D = D[selected_inds, :]
        indices = D.sum(axis=0).nonzero()[1]
    elif max_dist > self.max_dist:
        points_list = list(zip(self.initial_structure["nucleus.x"], self.initial_structure["nucleus.y"]))
        D = radius_neighbors_graph(points_list,
                                   max_dist,
                                   mode=mode,
                                   include_self=True,
                                   n_jobs=int(0.75 * multiprocessing.cpu_count()))
        D = D[selected_inds, :]
        indices = D.sum(axis=0).nonzero()[1]
    else:
        #         if cells_selected is None:
        D = self._graph_dict[selected_inds, :]
        indices = D.sum(axis=0).nonzero()[1]
        cells_selected = self.initial_structure.loc[self.initial_structure['index'].isin(indices)]
        cells_selected.reset_index(drop=True, inplace=True)
        cells_selected.index.name = 'new_index'
        cells_selected.reset_index(inplace=True)
        new_inds = cells_selected.loc[cells_selected['index'].isin(selected_inds)]['new_index']
        points_list = list(zip(cells_selected["nucleus.x"], cells_selected["nucleus.y"]))
        D = radius_neighbors_graph(points_list,
                                   max_dist,
                                   mode=mode,
                                   include_self=True,
                                   n_jobs=int(0.75 * multiprocessing.cpu_count()))
        D = D[new_inds, :]
        indices = D.sum(axis=0).nonzero()[1]
        cells_selected = cells_selected.loc[cells_selected['new_index'].isin(indices)]
        indices = cells_selected['index']

    return indices, D


def get_component_with_neighbors(self, column_name, component_number, max_dist=None):
    cells_comp = self.initial_structure.loc[self.initial_structure[column_name] == component_number]
    inds, D = get_cells_neighbors(self, cells_selected=cells_comp, max_dist=max_dist,
                                  index_col='index', mode='connectivity')
    comp_neighborhood = self.initial_structure.loc[self.initial_structure['index'].isin(inds)]
    comp_neighborhood.drop_duplicates(inplace=True)

    return comp_neighborhood