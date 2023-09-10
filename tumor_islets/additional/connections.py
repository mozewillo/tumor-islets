from scipy.spatial import distance
from sklearn.neighbors import radius_neighbors_graph
import multiprocessing
import pandas as pd
import numpy as np

def calculate_min_distances(self, celltype1='all', celltype2='all', max_dist=None, mapping=None,
                            save=False, directory=None):
    if max_dist is None:
        max_dist = self.max_dist
    if celltype1 != 'all':
        x0, y0, i0 = self.select_cells_by_celltype(celltype1)
    else:
        x0, y0, i0 = list(self.initial_structure['nucleus.x']), list(self.initial_structure['nucleus.y']), list(
            self.initial_structure['index'])
    if celltype2 != 'all':
        x1, y1, i1 = self.select_cells_by_celltype(celltype2)
    else:
        x1, y1, i1 = list(self.initial_structure['nucleus.x']), list(self.initial_structure['nucleus.y']), list(
            self.initial_structure['index'])
    mins = []
    ts = self._graph_dict[i0, :][:, i1]
    rows, cols = ts.nonzero()
    prev = 0
    for r, c in zip(rows, cols):
        if r != c:
            ts[r, c] = distance.euclidean((x0[r], y0[r]), (x1[c], y1[c]))
        else:
            ts[r, c] = 0
    for r in rows:
        rs = ts[r].todense()
        rs = rs[rs > 0]
        if rs.shape[1] > 0:
            mi = rs.min()
            mins.append(mi)
    mins = np.array(mins)
    mins = mins[mins < max_dist]
    if save:
        if directory is None:
            directory = self.working_directory
        np.save(f'{directory}min_dists_{max_dist}_{celltype1}_to_{celltype2}_{self.filename}.npy', mins)
    return mins


def count_neighbors(self, celltype1='all', celltype2='all', max_dist=None, mapping=None,
                    save=False, directory=None, mode='connectivity', in_tumor_ROI=None, min_thr=5):
    """
    For cells of celltype1 count number of celltype2 neighbors
    """
    if max_dist is None:
        max_dist = self.max_dist
    # select cells of given cell types

    # prepare neighborhood graph
    if in_tumor_ROI is not None:
        if in_tumor_ROI and 'in.ROI.tumor_tissue' in self.initial_structure.columns:
            roi_cells = self.initial_structure[self.initial_structure['in.ROI.tumor_tissue'] == True]
        elif not in_tumor_ROI and 'in.ROI.tumor_tissue' in self.initial_structure.columns:
            roi_cells = self.initial_structure[self.initial_structure['in.ROI.tumor_tissue'] == False]
        points_list = list(zip(roi_cells["nucleus.x"], roi_cells["nucleus.y"]))
        roi_cells.reset_index(inplace=True, drop=True)
        roi_cells.index.name = 'new.index'
        roi_cells.reset_index(inplace=True)
        i0 = roi_cells.loc[roi_cells['celltype'] == celltype1]['new.index']
        i1 = roi_cells.loc[roi_cells['celltype'] == celltype2]['new.index']
    else:
        points_list = list(zip(self.initial_structure["nucleus.x"], self.initial_structure["nucleus.y"]))
        x0, y0, i0 = self.select_cells_by_celltype(celltype1)
        x1, y1, i1 = self.select_cells_by_celltype(celltype2)

    if len(i0) < min_thr:
        return None, None

    if max_dist != self.max_dist or in_tumor_ROI is not None:
        G_temp = radius_neighbors_graph(points_list,
                                        max_dist,
                                        mode=mode,
                                        include_self=True,
                                        n_jobs=int(0.75 * multiprocessing.cpu_count()))
        G_n = G_temp[i0, :]
        G_temp = G_temp[i0, :][:, i1]
    else:
        G_temp = self._graph_dict[i0, :][:, i1]
        G_n = self._graph_dict[i0, :]

    counts = G_temp.sum(axis=1)
    if celltype1 == celltype2 or celltype1 == 'all' or celltype2 == 'all':
        counts = counts - 1  # (-1 because each cell is neighboring with itself)
    all_counts = G_n.sum(axis=1)  # calculate number of all neighbors of celltype1
    all_counts = all_counts - 1  # (-1 because each cell is neighboring with itself)
    if save:
        if directory is None:
            directory = self.working_directory
        np.save(f'{directory}counts_in_{max_dist}_{celltype1}_to_{celltype2}_{self.filename}.npy', counts)

    return counts, all_counts


def calculate_all_pairs_neighbor_counts(self, save=False, filename=None,
                                        mapping_file=None, max_dist=None, directory='./',
                                        in_tumor_ROI=None):
    # calculate this based on the on distance add simulatniously the mean value ect to summary matrix and save matrix
    if 'celltype' not in self.initial_structure.columns:
        self.map_phenotype_to_celltype(mapping=mapping_file)

    cts = self.initial_structure["celltype"].unique()
    neighbor_matrix = np.zeros((len(cts), len(cts)))
    freq_matrix = np.zeros((len(cts), len(cts)))
    mean_all_matrix = np.zeros((len(cts), len(cts)))

    if in_tumor_ROI == True or in_tumor_ROI == False:
        if 'in.ROI.tumor_tissue' not in self.initial_structure.columns:
            print('No ROI for the sample calculating for all cells')
            in_tumor_ROI = None

    for i, ct1 in enumerate(cts):
        for j, ct2 in enumerate(cts):
            c, ac = count_neighbors(self, ct1, ct2, max_dist=max_dist, save=False,
                                    directory=directory, mapping=mapping_file,
                                    in_tumor_ROI=in_tumor_ROI)
            if c is not None and ac is not None:
                mean_all_matrix[i][j] = c.sum() / ac.sum()
                ct1_n = len(self.initial_structure.loc[self.initial_structure['celltype'] == ct1])
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

    if save:
        if filename is None:
            filename = self.filename
        if in_tumor_ROI is None:
            suffix = f'{max_dist}__{filename}'
        elif in_tumor_ROI:
            suffix = f'in_tumor_{max_dist}__{filename}'
        elif not in_tumor_ROI:
            suffix = f'out_tumor_{max_dist}__{filename}'
        print(f'{directory}frac_neighboring_matrix_{suffix}.tsv')
        neighbor_matrix.to_csv(f'{directory}neighbors_matrix_neighbor_counts_{suffix}.tsv', sep='\t')
        freq_matrix.to_csv(f'{directory}frac_neighboring_matrix_{suffix}.tsv', sep='\t')
        mean_all_matrix.to_csv(f'{directory}neighbors_matrix_all_mean_{suffix}.tsv', sep='\t')

    return neighbor_matrix, freq_matrix, mean_all_matrix

def flatten_matrix(df, sample_id, columns_ordered, index_column='cell_type', take_nan=False):
    """Flatten matrix to vector, order by columns ordered to keep consistency with other samples"""
    possible_pairs = []
    i = 0
    dif = [i for i in columns_ordered if i not in df.columns] # check if all values were calulated
    if len(dif) > 0:
        print(f'{sample_id} : Missing columns: {dif}')
        if take_nan:
            for d in dif:
                df[d] = np.nan
        else:
            return None
    for c1 in columns_ordered:
        for c2 in columns_ordered:
            possible_pairs.append((c1, c2))
        if index_column not in df.columns and df.index.name == index_column:
            df.reset_index(inplace=True)
        df = df[[index_column] + columns_ordered]
        for i, ct in enumerate(columns_ordered):
            v = df.loc[df[index_column] == ct].iloc[0][1:].to_numpy()
            if i == 0:
                sample_vec = v
            else:
                sample_vec = np.hstack([sample_vec, v])

    return sample_vec


def celltype_connections_gather(files, mapping_file, index_column='cell_type', take_nan=True):
    """Gather results from given matrix files, flatten and concatenate into one matrix
        params: take_nan - boolean value - if True accept samples missing cell types
        and fill missing values with np.nan"""

    mapping = pd.read_csv(mapping_file)
    celltypes = list(mapping['celltype'].unique())
    possible_pairs = []
    samples_list = []

    for c1 in mapping['celltype'].unique():
        for c2 in mapping['celltype'].unique():
            possible_pairs.append((c1, c2))

    for fi, f in enumerate(files):
        sample_id = f.split('__')[-1].split('_#')[0]
        print(sample_id)
        cs = pd.read_csv(f, sep='\t')
        samples_list.append(sample_id)
        sample_vec = flatten_matrix(cs, sample_id, columns_ordered=celltypes,
                                    index_column=index_column, take_nan=take_nan)
        if fi == 0:
            all_samples = sample_vec
        else:
            all_samples = np.vstack([all_samples, sample_vec])

    all_samples = all_samples.astype('float')
    possible_pairs_str = [f'{p[0]}_{p[1]}' for p in possible_pairs]
    all_samples_df = pd.DataFrame(all_samples, index=samples_list, columns=possible_pairs_str)
    all_samples_df.index.name = 'sample_id'
    return all_samples_df