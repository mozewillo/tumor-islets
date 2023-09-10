import pathlib
import time
import os
import itertools
import json
import gc
import tqdm
import shapely.geometry
import matplotlib.pyplot as plt
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csgraph
from scipy import sparse
import pandas as pd
from collections import Counter
from collections import defaultdict
from scipy.spatial import distance
import tumor_islets.graph.graph_additional as Graph_additional
import tumor_islets.graph.concave_hull as Concave_Hull
from tumor_islets.graph.Cell import Cell, Marker
import numpy as np
import multiprocessing
import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback

panels_markers = {
    '1': ('CD11c', 'CD15', 'CD163', 'CD20', 'CD3', 'CK'),
    '2': ('CD8', 'CK', 'GB', 'Ki67', 'PD1', 'PDL1'),
    '3': ('CD3', 'CD4', 'CD56', 'CD8', 'CK', 'FOXP3')
}


class Graph(object):
    def __init__(self,
                 filename: str,
                 panel: str,
                 initial_structure: pd.DataFrame = None,
                 max_dist: float = 50,
                 mode: object = 'connectivity',
                 run_parallel: object = False,
                 include_all=True,
                 working_directory="./",
                 ):
        """
        Initializes a graph object.
        If no dictionary or None is given, an empty dictionary will be used
        :param filename: short identifier for the folder with results
        :param initial_structure: pandas dataframe from .tsv files
        :param max_dist: maximal distance between neighbouring cells (defaul = 50 microns, tried: 30)
        :param mode: possible mode in {‘connectivity’, ‘distance’} - CHECK
        :param run_parallel:
        :param include_all: if True we include all cells from initial_structure, otherwise
                            include only tumor cells by in.ROI.tumor_tissue (we didn't use that)
        """
        self._neighbor_graph = None
        self._tumor_tissue_graph = None
        self.max_dist = max_dist
        self.filename = filename
        # can't delete _#_cells - problems in SnakeMake :(
        # self.filename = filename.split("_#_cells")[0] # delete _#_cells_properties
        self.panel = panel
        self.initial_structure = initial_structure
        self.include_all = include_all
        # self._neighbour_graph = None  # graph for non CK
        self._id_to_position_mapping, self._position_to_cell_mapping = dict(), dict()
        # For connected components
        self._connected_components_positions = defaultdict(list)
        self._connected_components = None  # the output of connected components function
        # For invasive margins
        self._invasive_margins = defaultdict(list)
        self._invasive_margins_sequence = defaultdict(list)
        self._roi_margins = defaultdict(lambda: defaultdict(list))
        self.distance_matrix = None
        # IMPORTANT dicts
        self._position_to_component = dict()
        self._immune_position_to_component = defaultdict(lambda: -1)
        self._position_to_if_margin = defaultdict(lambda: False)
        self._position_to_invasive_margin = defaultdict(lambda: -1)

        self._position_to_margin = defaultdict(tuple)  # for margin slicing
        self._position_to_steepness = defaultdict(tuple)
        self.working_directory = working_directory + "Results/GRAPH_" + filename + "_dist_" + str(max_dist)
        # self.working_directory = working_directory + "Results/GRAPH_" + filename + "_dist_" + str(max_dist) + \
        #     "_include_" + str(include_all)

        markers, all_phenotypes = panels_markers[panel], []
        for comb in list(itertools.product(["+", "-"], repeat=6)):
            s = []
            for i, j in zip(markers, comb):
                s.append(i)
                s.append(j)
            all_phenotypes.append("".join(s))
        self.all_phenotypes = sorted(all_phenotypes, key=lambda x: x.count('CK-'))
        assert len(self.all_phenotypes) == 64

        if not os.path.exists(self.working_directory + "/graph_matrix.npz"):
            if not os.path.exists(self.working_directory):
                os.makedirs(self.working_directory)
            if not include_all and "in.ROI.tumor_tissue" in self.initial_structure.columns:
                initial_structure = initial_structure[initial_structure["in.ROI.tumor_tissue"] == "TRUE"]
            points_list = list(zip(initial_structure["nucleus.x"], initial_structure["nucleus.y"]))
            start = time.process_time()
            self._graph_dict = radius_neighbors_graph(points_list,
                                                      self.max_dist,
                                                      mode=mode,
                                                      include_self=True,
                                                      n_jobs=int(0.75 * multiprocessing.cpu_count()))
            print("[" + self.filename + "] Runtime, radius_neighbors_graph: " + str(time.process_time() - start))
            sparse.save_npz(self.working_directory + "/graph_matrix.npz", self._graph_dict)
            print("[" + self.filename + "] Radius neighbours graph matrix created and saved to:",
                  self.working_directory + "/graph_matrix.npz")
        else:
            self._graph_dict = sparse.load_npz(self.working_directory + "/graph_matrix.npz")
            print("[" + self.filename + "] Radius neighbours graph matrix read from file:",
                  self.working_directory + "/graph_matrix.npz")
        cx = self._graph_dict.tocoo()
        start = time.process_time()
        # unique_keys_id = {i for i in cx.row}
        unique_keys_id = self.initial_structure.reset_index()['index']
        # BASIC MULTIPROCESSING APPROACH
        if run_parallel:
            n = 20000
            idx_subsets = list(Graph_additional.divide_chunks(list(unique_keys_id), n))
            start = time.process_time()
            with multiprocessing.Pool(processes=int(0.75 * multiprocessing.cpu_count()),
                                      maxtasksperchild=1000) as pool:

                params = [(initial_structure, idx_subset) for idx_subset in idx_subsets]
                results = pool.starmap_async(Graph_additional.create_cells_parallel, params)
                for p in results.get():
                    tmp_id_to_position_mapping, tmp_position_to_cell_mapping = p
                    self._id_to_position_mapping.update(tmp_id_to_position_mapping)
                    self._position_to_cell_mapping.update(tmp_position_to_cell_mapping)
        # not parallel option
        else:
            for unique_key in unique_keys_id:
                cell_1 = Cell(self.initial_structure.iloc[unique_key], unique_key, self.panel)
                self._id_to_position_mapping[unique_key] = (cell_1.x, cell_1.y)
                self._position_to_cell_mapping[(cell_1.x, cell_1.y)] = cell_1
        gc.collect()
        self._position_to_id = {value: key for key, value in self._id_to_position_mapping.items()}
        print("[" + self.filename + "]Runtime, (Graph.__init__) {}, number of keys (cells) {}: ".format(
            time.process_time() - start,
            len(self._id_to_position_mapping)))
        self.initial_structure.reset_index(inplace=True)


    def select_cells_by_markers(self, positive_marker=None, negative_marker=None, marker_rule='and'):
        """
        Select cells expressing given markers
        Args:
            positive_marker: str/ list of positive markers for selection
            negative_marker: str/ list of negative markers for selection
            marker_rule: 'and' or 'or' logic rule for cell selection

        Returns:
            x, y, index for cells satisfying a given marker expression
        """
        if 'index' not in self.initial_structure.columns:
            self.initial_structure.reset_index(inplace=True)

        start_time = time.process_time()
        print("Start computing connected_components().")
        print(f'Selecting positive cells for markers: {positive_marker}')
        if positive_marker is not None:
            if type(positive_marker) == str:
                positive_marker = [positive_marker]
            positive_x, positive_y, idxs = [], [], []
            if marker_rule == 'or':
                for inx, m in enumerate(positive_marker):
                    if m + '.score.normalized' in self.initial_structure.columns:
                        if inx == 0:
                            positive = self.initial_structure.loc[
                                self.initial_structure[m + '.score.normalized'] > 1]
                        else:
                            pos = self.initial_structure.loc[self.initial_structure[m + '.score.normalized'] > 1]
                            positive = pd.concat([positive, pos])
                    positive = positive.drop_duplicates('index')
            elif marker_rule == 'and':
                print('SELECTING..')
                for inx, m in enumerate(positive_marker):
                    if inx == 0:
                        if m + '.score.normalized' in self.initial_structure.columns:
                            positive = self.initial_structure.loc[
                                self.initial_structure[m + '.score.normalized'] > 1]
                    else:
                        if m + '.score.normalized' in self.initial_structure.columns:
                            positive = positive.loc[positive[m + '.score.normalized'] > 1]

        positive_x, positive_y, idxs = positive['nucleus.x'].to_numpy(), positive['nucleus.y'].to_numpy(), positive['index'].to_numpy()

        return positive_x, positive_y, idxs

    def map_phenotype_to_celltype(self, mapping=None):
        """Map marker phenotype to cell types based on mapping
        Args:
            mapping: [optional] path to csv file with columns 'phenotype' and 'celltype' corresponding to custom mapping
            """
        this_dir, this_filename = os.path.split(__file__)
        this_dir = '/'.join(this_dir.split('/')[:-1])

        if mapping is not None:
            mapping = pd.read_csv(mapping)
        elif self.panel == '1' or self.panel == 'IF1':
            mapping = pd.read_csv(f'{this_dir}/data/IF1_phen_to_cell_mapping.csv')
        elif self.panel == '2' or self.panel == 'IF2':
            mapping = pd.read_csv('../data/IF2_phen_to_cell_mapping.csv')
        elif self.panel == '3' or self.panel == 'IF3':
            mapping = pd.read_csv('../data/IF3_phen_to_cell_mapping_initial_64.csv')
        else:
            print(f'Custom panel {self.panel}, provide custom mapping')
            return 1
        # order markers in phenotype column same as in initial dataframe
        phs = self.initial_structure['phenotype'].unique()
        phs.sort()
        marker_order = [m for p in phs[-1].split('+') for m in p.split('-')]
        original = []
        reordered = []
        for ph in mapping['phenotype'].unique():
            ph_str = ''
            for m in marker_order:
                if m != '':
                    ph_str += m + ph.split(m)[-1][0]
            original.append(ph)
            reordered.append(ph_str)
        ph_dict = {'phenotype': original, 'phenotype_reordered': reordered}
        ph_dict = pd.DataFrame.from_dict(ph_dict)
        mapping = pd.merge(mapping, ph_dict, on='phenotype')
        mapping = mapping[['celltype', 'phenotype_reordered']]
        mapping.columns = ['celltype', 'phenotype']

        self.initial_structure = pd.merge(self.initial_structure, mapping, on='phenotype', how='left')
        return 0

    def get_cells_neighbors(self, cells_selected=None, selected_inds=None,
                            max_dist=None, index_col='index', mode='connectivity'):
        """for given cells (rows from df) or selected indices find all neighbors in given distance
            Returns: neighboring cells indices and selected neighborhood graph"""
        if cells_selected is not None and selected_inds is None:
            selected_inds = list(cells_selected['index'])
        elif selected_inds is None and cells_selected is None:
            print('No cells selected')
            return None
        if max_dist is None:
            D = self._graph_dict
        else:
            points_list = list(zip(self.initial_structure["nucleus.x"], self.initial_structure["nucleus.y"]))
            D = radius_neighbors_graph(points_list,
                                       max_dist,
                                       mode=mode,
                                       include_self=True,
                                       n_jobs=int(0.75 * multiprocessing.cpu_count()))
        D = D[selected_inds, :]
        indices = D.sum(axis=0).nonzero()[1]
        return indices, D

    def get_roi_margin(self, roi_column='in.ROI.tumor_tissue', alpha=30, get_neighbors=True,
                       max_dist=None, neighbors_dist=None, add_to_df=True, min_margin_size=100):
        neighbors_all = None
        margins = None
        new_cols = []
        print(f'["{self.filename}"] Using cells with: {roi_column}=True')
        tumor_cells = self.initial_structure.loc[self.initial_structure[roi_column] == True]
        cell_coords = list(zip(tumor_cells['nucleus.x'], tumor_cells['nucleus.y']))
        #
        if max_dist is None: max_dist = self.max_dist
        tumor_tissue_graph = radius_neighbors_graph(cell_coords, max_dist)
        tumor_components = csgraph.connected_components(tumor_tissue_graph)
        tumor_cells[f'{roi_column}.component_number'] = tumor_components[1]
        new_cols.append(f'{roi_column}.component_number')
        print("Calculating margins")
        n = 0
        for number in range(tumor_components[0]):
            subset = tumor_cells.loc[tumor_cells[f'{roi_column}.component_number'] == number]
            coords = list(zip(subset['nucleus.x'], subset['nucleus.y']))
            sequence, points = self.determine_margin_tumor(coords, alpha)
            if sequence is None or len(sequence) < min_margin_size:
                continue
            n += 1
            print(f'Margin {number}: {len(sequence)} cells')
            in_margin = pd.DataFrame(points, columns=['nucleus.x', 'nucleus.y'])
            in_margin[f'{roi_column}.margin'] = n  # margins will be numerated starting from 1
            # get neighbors of margin
            if n == 1:
                margins = in_margin
                new_cols.append(f'{roi_column}.margin')
            else:
                margins = pd.concat([margins, in_margin])
        if margins is not None:
            tumor_cells = pd.merge(tumor_cells, margins, on=['nucleus.x', 'nucleus.y'], how='left')
        if f'{roi_column}.margin' not in tumor_cells.columns:
            get_neighbors = False

        n = 0
        if get_neighbors:
            print('Add margin neighbors')
            for number in tumor_cells[f'{roi_column}.margin'].unique():
                margin_cells = tumor_cells.loc[tumor_cells[f'{roi_column}.margin'] == number]
                if len(margin_cells) == 0:
                    continue
                n_inds, D= self.get_cells_neighbors(cells_selected=margin_cells, max_dist=neighbors_dist)
                neighbors = self.initial_structure.loc[self.initial_structure['index'].isin(n_inds)]
                neighbors[f'{roi_column}.neighbor'] = number
                print(f'Margin {number}: {len(neighbors)} neighboring cells')
                n += 1
                if n == 1:
                    neighbors_all = neighbors
                    new_cols.append(f'{roi_column}.neighbor')
                else:
                    neighbors_all = pd.concat([neighbors_all, neighbors])

        if neighbors_all is not None:
            neighbors_all.drop_duplicates('index', inplace=True)
            # omitt cells that are already in margins
            inds = list(tumor_cells.loc[~tumor_cells[f'{roi_column}.margin'].isnull(), 'index'])
            neighbors_all = neighbors_all.loc[~neighbors_all['index'].isin(inds)]
            # add neighbors in roi to tumor cells
            in_roi = neighbors_all.loc[neighbors_all[roi_column] == True]
            if len(in_roi) > 0:
                in_roi = in_roi[['index', f'{roi_column}.neighbor']]
                tumor_cells = pd.merge(tumor_cells, in_roi, on='index', how='left')
            # add not in roi cells
            not_in_roi = neighbors_all.loc[neighbors_all[roi_column] == False]
            if len(not_in_roi) > 0:
                tumor_cells = pd.concat([tumor_cells, not_in_roi])

        for m in tumor_cells[f'{roi_column}.margin'].unique():
            tumor_cells.loc[tumor_cells[f'{roi_column}.margin'] == m, f'{roi_column}.full_margin'] = m
            if f'{roi_column}.neighbor' in tumor_cells.columns:
                tumor_cells.loc[tumor_cells[f'{roi_column}.neighbor'] == m, f'{roi_column}.full_margin'] = m
        new_cols.append(f'{roi_column}.full_margin')
        tumor_cells = tumor_cells.drop_duplicates()
        if add_to_df:
            new_cols = [ncol for ncol in new_cols if ncol in tumor_cells.columns]
            cols = ['index'] + new_cols
            tumor_cells_tmp = tumor_cells[cols]
            self.initial_structure = pd.merge(self.initial_structure, tumor_cells_tmp, on='index', how='left')

        return tumor_cells

    def get_ROI_cells_plus_neighbors(self, in_ROI_column='in.ROI.tumor_tissue',
                                     max_dist=None, mode='connectivity'):
        inds = list(self.initial_structure.loc[self.initial_structure[in_ROI_column] == True, 'index'])
        if max_dist == self.max_dist or max_dist is None:
            D = self._graph_dict
        else:
            points_list = list(zip(self.initial_structure["nucleus.x"], self.initial_structure["nucleus.y"]))
            D = radius_neighbors_graph(points_list,
                                       max_dist,
                                       mode=mode,
                                       include_self=True,
                                       n_jobs=int(0.75 * multiprocessing.cpu_count()))
        indices = D[inds, :].sum(axis=0).nonzero()[1]
        self.initial_structure.loc[
            self.initial_structure['index'].isin(indices), f'{in_ROI_column}.plus_neighborhood'] = 1

    def select_cells_by_celltype(self, celltype, mapping=None):
        """Select cells of given cell type"""
        if 'celltype' not in self.initial_structure.columns:
            self.map_phenotype_to_celltype(mapping=mapping)
        # if 'all' return all cells x,y coordinates and indices
        if celltype == 'all':
            return self.initial_structure['nucleus.x'].to_numpy(), self.initial_structure['nucleus.y'].to_numpy(), \
                self.initial_structure['index'].to_numpy()
        # if invalid celltype name return empty lists
        elif celltype not in self.initial_structure['celltype'].unique():
            print(f'No celltype {celltype} in mapped cells. Celltypes available:'
                  f' {self.initial_structure["celltype"].unique()}')
            return [], [], []
        # return selected celltype x,y coordinates and indices
        selected = self.initial_structure.loc[self.initial_structure['celltype'] == celltype]
        return selected['nucleus.x'].to_numpy(), selected['nucleus.y'].to_numpy(), selected['index'].to_numpy()

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
                        save=False, directory=None, mode='connectivity'):
        """
        For cells of celltype1 count number of celltype2 neighbors
        """
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

        if max_dist != self.max_dist:
            if celltype1 == celltype2:
                points_list = list(zip(x0, y0))
            elif celltype1 == 'all' or celltype2 == 'all':
                points_list = list(zip(self.initial_structure["nucleus.x"], self.initial_structure["nucleus.y"]))

            else:
                xs = x0.extend(x1)
                ys = y0.extend(y1)
                points_list = list(zip(xs, ys))
            G_temp = radius_neighbors_graph(points_list,
                                            max_dist,
                                            mode=mode,
                                            include_self=True,
                                            n_jobs=int(0.75 * multiprocessing.cpu_count()))
        else:
            G_temp = self._graph_dict[i0, :][:, i1]
        counts = G_temp.sum(axis=1)
        if save:
            if directory is None:
                directory = self.working_directory
            np.save(f'{directory}counts_in_{max_dist}_{celltype1}_to_{celltype2}_{self.filename}.npy', counts)
        return counts


    def select_cells_by_numeric_value(self, numeric_column, numeric_threshold, above=True):
        """
        Args:
            numeric_column: columns numerical with numerical values on which the selection is based on
            numeric_threshold: threshold for the selected numerical column values
            above: if true select values above the threshold else below the threshold
        Returns: x, y coordinates, and indices of cells with the numerical value above or below the given threshold

        """
        if numeric_column not in self.initial_structure:
            print(f"Wrong column name, no column named: {numeric_column}")
            return [], [], []
        if above:
            cells = self.initial_structure.loc[self.initial_structure[numeric_column] > numeric_threshold]
        else:
            cells = self.initial_structure.loc[self.initial_structure[numeric_column] < numeric_threshold]

        return cells['nucleus.x'].to_numpy(), cells['nucleus.y'].to_numpy(), cells['index'].to_numpy()

    def calculate_marker_columns(self):
        markers = [c.split('.score')[0] for c in self.initial_structure.columns if c.endswith('.score.normalized')]
        for m in markers:
            self.initial_structure.loc[self.initial_structure[f'{m}.score.normalized'] > 1, f'{m}.positive'] = 1
            self.initial_structure.loc[self.initial_structure[f'{m}.score.normalized'] <= 1, f'{m}.positive'] = 0

    def calculate_marker_homogeneity_score(self, positive_marker, max_dist=None, mode='connectivity'):
        f"""
        For each cell expressing <positive_marker> calculates number of neighboring cells expressing <positive_marker>
        divided by total number of neighbors
        Args:
            positive_marker: marker for whitch to calculate homogeneity score

        Returns: Adds column to intial structure named '<positive_marker>.homogenic.connections.score'

        """
        if f'{positive_marker}.positive' not in self.initial_structure.columns:
            self.calculate_marker_columns()

        inds = list(self.initial_structure.loc[self.initial_structure[f'{positive_marker}.positive'] == 1, 'index'])
        if max_dist == self.max_dist or max_dist is None:
            D = self._graph_dict
        else:
            points_list = list(zip(self.initial_structure["nucleus.x"], self.initial_structure["nucleus.y"]))
            D = radius_neighbors_graph(points_list,
                                       max_dist,
                                       mode=mode,
                                       include_self=True,
                                       n_jobs=int(0.75 * multiprocessing.cpu_count()))
        D = D[inds, :]
        n_neighbors = np.array(D.sum(axis=1).flatten())[0]
        Di = D[:, inds]
        pos = np.array(Di.sum(axis=1).flatten())[0]
        neg = n_neighbors - pos
        connections = pd.DataFrame(list(zip(inds, n_neighbors, pos, neg)), columns=['index', 'n_neighbors',
                                                                                    f'{positive_marker}.homogenic',
                                                                                    f'{positive_marker}.heterogenic'])
        connections[f'{positive_marker}.homogenic.connections.score'] = [x / y if y > 0 else None
                                                                         for x, y in
                                                                         zip(connections[
                                                                                 f'{positive_marker}.homogenic'],
                                                                             connections['n_neighbors'])]
        self.initial_structure = pd.merge(self.initial_structure, connections, on='index', how='left')

    def calculate_marker_connection_score(self, marker1, marker2, max_dist, mode='connectivity'):
        f"""
        For each cell expressing <positive_marker> calculates number of neighboring cells expressing <positive_marker>
        divided by total number of neighbors
        Args:
            positive_marker: marker for whitch to calculate homogeneity score

        Returns: Adds column to intial structure named '<positive_marker>.homogenic.connections.score'

        """
        if f'{marker1}.positive' not in self.initial_structure.columns:
            self.calculate_marker_columns()

        inds = list(self.initial_structure.loc[self.initial_structure[f'{marker1}.positive'] == 1, 'index'])
        inds2 = list(self.initial_structure.loc[self.initial_structure[f'{marker2}.positive'] == 1, 'index'])
        if max_dist == self.max_dist or max_dist is None:
            D = self._graph_dict
        else:
            points_list = list(zip(self.initial_structure["nucleus.x"], self.initial_structure["nucleus.y"]))
            D = radius_neighbors_graph(points_list,
                                       max_dist,
                                       mode=mode,
                                       include_self=True,
                                       n_jobs=int(0.75 * multiprocessing.cpu_count()))
        D = D[inds, :]
        n_neighbors = np.array(D.sum(axis=1).flatten())[0]
        Di = D[:, inds2]
        pos = np.array(Di.sum(axis=1).flatten())[0] / n_neighbors
        connections = pd.DataFrame(list(zip(inds, pos)), columns=['index', f'{marker1}_to_{marker2}'])
        self.initial_structure = pd.merge(self.initial_structure, connections, on='index', how='left')

    def calculate_isolated_cells(self, positive_marker='CK', negative_markers=['CD3', 'CD20'],
                                 column_name='No.lymphocytes',
                                 threshold=0.0, max_dist=None):
        if positive_marker is None:
            positive_marker = 'all'

        if negative_markers is None:
            if self.panel == '1' or self.panel == 'IF1':
                negative_markers = ['CD3', 'CD20']
            elif self.panel == '2' or self.panel == 'IF2':
                negative_markers = ['CD8']
            elif self.panel == '3' or self.panel == 'IF3':
                negative_markers = ['CD3', 'CD8', 'CD4']

        for i, m in enumerate(negative_markers):
            if f'{positive_marker}_to_{m}' not in self.initial_structure:
                self.calculate_marker_connection_score(positive_marker, m, max_dist=max_dist)
            ind = self.initial_structure.loc[self.initial_structure[f'{positive_marker}_to_{m}'] <= threshold, 'index']
            if i == 0:
                inds = ind
            else:
                inds = set(ind).intersection(set(inds))
        self.initial_structure.loc[self.initial_structure['index'].isin(inds), column_name] = 1

    def calculate_number_of_neighbors(self):
        n_neighbors = np.array(self._graph_dict.sum(axis=1).flatten())[0]
        self.initial_structure['n.neighbors'] = n_neighbors

    def connected_components(self, positive_marker=None, negative_marker=None, celltype=None, numeric_column=None,
                             numeric_threshold=None, mode='connectivity',
                             save_df_components=False, max_dist=None, directory=None, marker_rule='and',
                             add_to_summary=True, summary_dir=None):
        """
        Compute connected components on CK cells.
        Creates a dictionary self._new_position_to_component = {pos:component number}
        where value -1 is given to all CK negatives cells.
        Args:
        max_dist: maximal distance for two neighboring cells in connected component
        negative_marker:
        positive_marker:
        directory: write results in different directory than default
        save_df_components: if True then the function saves an original dataframe with
                            an additional component_number column to
        mode: possible mode in {‘connectivity’, ‘distance’} - CHECK

        :return: a tuple, where the first element is the number of components,
        and an array of [-1, 2, 3,..., 30] with component numbers for corresponding cell ids.
        """

        if save_df_components:
            if directory is None:
                directory = self.working_directory
            else:
                if not os.path.exists(directory):
                    os.makedirs(directory)
        if add_to_summary:
            if summary_dir is None:
                summary_dir = self.working_directory

        if positive_marker is None and negative_marker is None and celltype is None and numeric_column is None:
            positive_marker = 'CK'

        if numeric_column is not None:
            if numeric_threshold is None:
                numeric_threshold = 1
            positive_x, positive_y, idxs = self.select_cells_by_numeric_value(numeric_column, numeric_threshold)
            marker_str = numeric_column

        elif positive_marker is not None or negative_marker is not None:
            if type(positive_marker) == str:
                positive_marker = [positive_marker]
            positive_x, positive_y, idxs = self.select_cells_by_markers(positive_marker=positive_marker,
                                                                        negative_marker=negative_marker,
                                                                        marker_rule=marker_rule)
            marker_str = '_'.join(positive_marker)

        elif celltype is not None:
            positive_x, positive_y, idxs = self.select_cells_by_celltype(celltype)
            marker_str = celltype

        # if no positive cells write file empty dataframe if save mode, return component_number -1 for all cells
        if len(idxs) == 0:
            print("No cells found")
            self._connected_components = -1
            if save_df_components:
                print("No cells write empty file and pass")
                dataframe_copy = self.initial_structure.copy()[['index', 'cell.ID', 'nucleus.x', 'nucleus.y']]
                dataframe_copy[f"{marker_str}.component_number"] = -1
                dataframe_copy.to_csv(
                    f'{directory}/df_with_components_{max_dist}_{marker_str}_{marker_rule}_{str(self.filename)}.tsv',
                    sep='\t')
            return self._connected_components

        positive_cells = list(zip(positive_x, positive_y))

        start_time = time.process_time()
        print("Start computing connected_components().")
        print(f'Selecting positive cells for markers: {positive_marker}')
        print(f'{len(positive_cells)} positive cells selected.')

        if max_dist is None:
            max_dist = self.max_dist

        self._neighbor_graph = radius_neighbors_graph(positive_cells,
                                                      max_dist,
                                                      mode=mode,
                                                      include_self=False,
                                                      n_jobs=int(0.75 * multiprocessing.cpu_count()))
        # Compute connected components
        self._connected_components = csgraph.connected_components(self._neighbor_graph)

        print("\n[" + self.filename + "] The largest components:\n",
              Counter(self._connected_components[1]).most_common(5))

        tmp = np.array([-1] * len(self._id_to_position_mapping.keys()))
        tmp[np.array(idxs)] = self._connected_components[1]
        self.initial_structure[f"{marker_str}.component_number"] = tmp

        # Save results to tsv
        if save_df_components:
            dataframe_copy = self.initial_structure.copy()[['index', 'cell.ID', 'nucleus.x', 'nucleus.y']]
            dataframe_copy[f"{marker_str}.component_number"] = tmp
            dataframe_copy.to_csv(
                f'{directory}/df_with_components_{max_dist}_{marker_str}_{marker_rule}_{str(self.filename)}.tsv',
                sep='\t')
            del dataframe_copy
            gc.collect()

        if add_to_summary:
            dataframe_copy = self.initial_structure.copy()
            if os.path.exists(f'{summary_dir}/all_data_{str(self.filename)}.tsv'):
                summary = pd.read_csv(f'{summary_dir}/all_data_{str(self.filename)}.tsv', sep='\t')
                dataframe_copy = dataframe_copy[['cell.ID']]
                dataframe_copy[f"{marker_str}.component_number"] = tmp
                summary = pd.merge(summary, dataframe_copy, on='cell.ID', how='left')
                summary.to_csv(f'{summary_dir}/all_data_{str(self.filename)}.tsv', sep='\t')
            else:
                dataframe_copy[f"{marker_str}.component_number"] = tmp
                dataframe_copy.to_csv(f'{summary_dir}/all_data_{str(self.filename)}.tsv', sep='\t')
            del dataframe_copy
            gc.collect()

        self._connected_components = (self._connected_components[0], tmp)
        print("[" + self.filename + "] Number of components found:", self._connected_components[0])
        mid_time = time.process_time()

        self._connected_components_positions = {i: [self._id_to_position_mapping[id]
                                                    for id in np.where(self._connected_components[1] == i)[0]]
                                                for i in range(-1, self._connected_components[0])}

        self._position_to_component = {value: key
                                       for key, l in self._connected_components_positions.items()
                                       for value in l}
        gc.collect()
        print("[" + self.filename + "] Runtime, dictionary connected_components(): {}. ".format(
            round(time.process_time() - mid_time, 3)))
        print("[" + self.filename + "] Runtime, connected_components(): {}. ".format(
            round(time.process_time() - start_time, 3)))

        return self._connected_components

    def save_to_tsv(self, summary_dir=None):
        if summary_dir is None:
            summary_dir = self.working_directory
        self.initial_structure.to_csv(f'{summary_dir}/all_data_{str(self.filename)}.tsv', sep='\t')

    def roi_margins_layered(self, alpha=30, in_levels=-3, out_levels=3, recompute=True):
        available_rois = [name for name in self.initial_structure.columns if "ROI" in name]
        if self._roi_margins and not recompute:
            return self._roi_margins
        else:
            # At this point either:
            # - self._roi_margins == {}
            # - the request is to recompute
            for roi_column in available_rois:
                self.initial_structure[roi_column + "_margin"] = None
                roi_data = self.initial_structure[roi_column]
                cells_ids = [idx for idx, val in enumerate(roi_data) if val]
                if len(cells_ids) > 10:
                    points_for_concave_hull = [self._id_to_position_mapping[id] for id in cells_ids]
                    level_in_cells = self.roi_margins_level(points_for_concave_hull, roi_column,
                                                            level=0, alpha=alpha, recompute=recompute)
                    level_out_cells = level_in_cells.copy()
                    print(cells_ids)
                    print(level_in_cells)
                    points_for_concave_hull_IN = points_for_concave_hull.copy()
                    points_for_concave_hull_OUT = points_for_concave_hull.copy()
                    # compute inner layers of margin
                    for in_level in range(1, -in_levels + 1):
                        points_for_concave_hull_IN = [pos for pos in points_for_concave_hull_IN if
                                                      pos not in level_in_cells]
                        if len(points_for_concave_hull_IN) > 10:
                            # points_for_concave_hull_IN = [self._id_to_position_mapping[id] for id in cells_ids]
                            level_in_cells = self.roi_margins_level(points_for_concave_hull_IN, roi_column,
                                                                    level=-in_level, alpha=alpha, recompute=recompute)
                    print(level_out_cells)
                    print("Compute outter layers")
                    for out_level in range(1, out_levels + 1):
                        for pos in level_out_cells:
                            cell_id = self._position_to_id[pos]
                            neigh_indices = self._graph_dict[cell_id, :].indices
                            new_levels = [out_level if not x else x for x in
                                          self.initial_structure.loc[neigh_indices, roi_column + "_margin"]]
                            self.initial_structure.loc[neigh_indices, roi_column + "_margin"] = new_levels
                            neigh_positions = [self._id_to_position_mapping[idx] for idx in
                                               self._graph_dict[cell_id, :].indices]
                            points_for_concave_hull_OUT.extend(neigh_positions)

                        points_for_concave_hull_OUT = list(dict.fromkeys(points_for_concave_hull_OUT))
                        level_out_cells = self.roi_margins_level(points_for_concave_hull_OUT, roi_column,
                                                                 level=out_level, alpha=alpha, recompute=recompute)
        return self._roi_margins

    def roi_margins_level(self, points_for_concave_hull, roi_column, level=0, alpha=30, recompute=True):
        concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
        if isinstance(concave_hull, shapely.geometry.Polygon):
            margin_edge_sequence, margin_positions, _ = Graph_additional._determine_margin_helper(
                boundary=concave_hull.boundary, number=-2)
            for pos in margin_positions:
                if self.initial_structure[roi_column + "_margin"][self._position_to_id[pos]] is None:
                    self.initial_structure.loc[self._position_to_id[pos], roi_column + "_margin"] = level
            self._roi_margins[roi_column][level].extend(margin_positions)
        else:
            for geom in concave_hull.geoms:
                margin_edge_sequence, margin_positions, _ = Graph_additional._determine_margin_helper(
                    boundary=geom.boundary, number=-2)
                for pos in margin_positions:
                    if self.initial_structure[roi_column + "_margin"][self._position_to_id[pos]] is None:
                        self.initial_structure.loc[self._position_to_id[pos], roi_column + "_margin"] = level

                # self.initial_structure[roi_column + "_margin"] = level
                self._roi_margins[roi_column][level].extend(margin_positions)
            # self.characterize_roi_margin(roi_column)
            if level > 0:
                unique_positions = [pos for pos in self._roi_margins[roi_column][level]
                                    if pos not in self._roi_margins[roi_column][level - 1]]
                self._roi_margins[roi_column][level] = unique_positions
        return self._roi_margins[roi_column][level]

    def roi_margins(self, level=0, alpha=30, recompute=True):
        available_rois = [name for name in self.initial_structure.columns if "ROI" in name]
        if self._roi_margins and not recompute:
            return self._roi_margins
        else:
            # At this point either:
            # - self._roi_margins == {}
            # - the request is to recompute
            for roi_column in available_rois:
                roi_data = self.initial_structure[roi_column]
                cells_ids = [idx for idx, val in enumerate(roi_data) if val]
                if len(cells_ids) > 10:
                    points_for_concave_hull = [self._id_to_position_mapping[id] for id in cells_ids]
                    concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
                    if isinstance(concave_hull, shapely.geometry.Polygon):
                        margin_edge_sequence, margin_positions, _ = Graph_additional._determine_margin_helper(
                            boundary=concave_hull.boundary, number=-2)
                        self._roi_margins[roi_column][level].extend(margin_positions)
                    else:
                        for geom in concave_hull.geoms:
                            margin_edge_sequence, margin_positions, _ = Graph_additional._determine_margin_helper(
                                boundary=geom.boundary, number=-2)
                            self._roi_margins[roi_column][level].extend(margin_positions)
                    self.characterize_roi_margin(roi_column)
            return self._roi_margins

    def characterize_roi_margin(self, roi_name, plot_labels=False, save_plot=True,
                                display_plots=True):
        directory = self.working_directory + "/ROIs/" + roi_name
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        all_roi_phenotypes = []
        inv_mar = [self._position_to_cell_mapping[position].phenotype_label
                   for position in self._roi_margins[roi_name]]
        all_roi_phenotypes.extend(inv_mar)
        roi_label = [roi_name] * len(inv_mar)
        data = {'ROI label': roi_label, 'Cell Type': all_roi_phenotypes}
        pal = Graph_additional._create_color_dictionary(set(all_roi_phenotypes))
        df = pd.DataFrame(data)
        ax = pd.crosstab(df['ROI label'], df['Cell Type']).apply(lambda r: r / r.sum() * 100, axis=1)
        tmp = pd.crosstab(df['ROI label'], df['Cell Type'])
        tmp.to_csv(directory + "/roi-margin-description-{}.csv".format(self.max_dist))
        if save_plot:
            total_cells = tmp.transpose().sum().tolist()
            ax_1 = ax.plot.bar(figsize=(16, 9), stacked=True, rot=0, color=pal)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       fancybox=True, shadow=True, ncol=7, title="Gene Marker")
            plt.xlabel('ROI label')
            plt.ylabel('Percent Distribution')
            for idx, rec in enumerate(ax_1.patches):
                height = rec.get_height()
                _row = idx % tmp.shape[0]
                _col = idx // tmp.shape[0]
                if plot_labels:
                    ax_1.text(rec.get_x() + rec.get_width() / 2,
                              rec.get_y() + height / 2,
                              "{:.0f}%".format(height) if height >= 5 else " ",
                              ha='center',
                              va='bottom',
                              rotation=90)
            labels = [item.get_text() + " (" + str(total_cells[idx]) + ")" for idx, item in
                      enumerate(ax_1.get_xticklabels())]
            ax_1.set_xticklabels(labels)
            plt.xticks(rotation=90)
            plt.savefig(directory + "/roi-margin-description-{}.pdf".format(self.max_dist))
            if display_plots:
                plt.show()

    ##############################################
    # FOR CLUSTERING
    ##############################################
    def cluster_tumor_islets_interior(self, directory=None):
        """
        For each component the function saves positions of all cells within component to the specific
        folder in the .npy file.
        Then it computes the distribution of markers inside the component, and in the neighbourhood
        of the component, saves 2 csv files:
        - cluster_tumor_islets_interior - phenotypes of cells inside the component (only CK+),
        - cluster_tumor_islets_full_interior - phenotypes of cells inside and in the neighbourhood
        of the component (also non CK+ cells)
        :return:
        """

        if directory is None:
            directory = self.working_directory
        else:
            if directory.endswith(f'GRAPH_{self.filename}/'):
                directory = directory + f'GRAPH_{self.filename}/'
        print("[" + self.filename + "] Start computing self.cluster_tumor_islets_interior()")
        if not self._connected_components:
            self.connected_components()
        df_interior = pd.DataFrame(columns=["filename", "component_number", "max_dist"] + self.all_phenotypes)
        df_full_interior = df_interior.copy()
        index_i, index_o = 0, 0
        if not os.path.exists(directory + "Components"):
            os.makedirs(directory + "Components")
        for component_number in tqdm.tqdm(range(self._connected_components[0])):
            directory = directory + "Components/component_" + str(component_number)
            if not os.path.exists(directory):
                os.makedirs(directory)
            phen = {"positions": self._connected_components_positions[component_number],
                    "phenotypes": [self._position_to_cell_mapping[tuple(p)].phenotype_original
                                   for p in self._connected_components_positions[component_number]]}
            if len(phen["positions"]) > 1000:
                n = len(phen["positions"])
                print(f"[{self.filename}] Component with {n} cells, component number = {component_number}")
            if len(phen["positions"]) > 50:  # analyze big enough components
                count_in = Counter(phen["phenotypes"])
                for key, value in count_in.items():
                    df_interior.loc[index_i, key] = value
                    df_interior.loc[index_i, "component_number"] = component_number
                index_i += 1
                # find phenotypes of neighbouring cells
                add_cells = [(i, j) for i, j in zip(phen["positions"], phen["phenotypes"])]
                for pos in self._connected_components_positions[component_number]:
                    neigh_id = self._position_to_id[pos]
                    for j in self._graph_dict[neigh_id, :].indices:
                        pos_j = self._id_to_position_mapping[j]
                        phen_j = self._position_to_cell_mapping[pos_j].phenotype_original
                        add_cells.append((pos_j, phen_j))
                        if "CK-" in phen_j:
                            self._immune_position_to_component[pos_j] = component_number
                add_cells = list(set(add_cells))
                JsonString = json.dumps({"positions": [i[0] for i in add_cells],
                                         "phenotypes": [i[1] for i in add_cells]})
                new_phen = pd.DataFrame({"positions": [i[0] for i in add_cells],
                                         "phenotypes": [i[1] for i in add_cells]})
                count_out = Counter(new_phen["phenotypes"])
                for key, value in count_out.items():
                    df_full_interior.loc[index_o, key] = value
                    df_full_interior.loc[index_o, "component_number"] = component_number
                index_o += 1
                f = open(directory + "component_" + str(component_number) + "_all_pos.json", "w")
                f.write(JsonString)
                f.close()
                gc.collect()
        print("[" + self.filename + "] Computed all islets interiors and saved to all_pos.npy files.")
        df_interior = df_interior.fillna(0)
        df_full_interior = df_full_interior.fillna(0)
        df_interior["filename"] = self.filename
        df_interior["max_dist"] = self.max_dist
        df_full_interior["filename"] = self.filename
        df_full_interior["max_dist"] = self.max_dist
        df_interior.to_csv(directory + "cluster_tumor_islets_interior.csv",
                           index=False, header=True)
        df_full_interior.to_csv(directory + "cluster_tumor_islets_full_interior.csv",
                                index=False, header=True)
        gc.collect()
        return df_interior, df_full_interior

    def cluster_tumor_islets_boundaries(self, alpha=30, directory=None):
        """
        Determine margins of all components and create a dataframe that summarizes the
        distribution of phenotypes in each component (that has a boundary).
        If a component has probably less than 10 poits it has no boundary.
        :param alpha: parameter for concave hull detection
        :return: pd dataframe with counts of cell phenotypes on the boundary of each component
        that has a nonempty boundary
        """
        if directory is None:
            directory = self.working_directory
        else:
            if directory.endswith(f'GRAPH_{self.filename}/'):
                directory = directory + f'GRAPH_{self.filename}/'

        print("[" + self.filename + "] Start computing self.cluster_tumor_islets_boundaries()")
        self.determine_all_margins(alpha=alpha)
        column_names = ["filename", "component_number", "max_dist", "alpha"] + self.all_phenotypes
        df_boundaries = pd.DataFrame(columns=column_names)
        index = 0
        which_components = [n for n in list(range(self._connected_components[0]))
                            if len(self._connected_components_positions[n]) > 50]
        for component_number in which_components:
            boundary = self._invasive_margins[component_number]
            if boundary:
                for p in boundary:
                    self._position_to_if_margin[tuple(p)] = True
                phen = [self._position_to_cell_mapping[tuple(p)].phenotype_original
                        for p in boundary]
                count_phen = Counter(phen)
                for key, value in count_phen.items():
                    df_boundaries.loc[index, key] = value
                    df_boundaries.loc[index, "component_number"] = component_number
            index += 1
        gc.collect()
        print("[" + self.filename + "] Successfully computed all margin boundaries.")
        df_boundaries["filename"] = self.filename
        df_boundaries["max_dist"] = self.max_dist
        df_boundaries["alpha"] = alpha
        df_boundaries = df_boundaries.fillna(0)
        df_boundaries.to_csv(directory + "cluster_tumor_islets_boundaries.csv", index=False, header=True)
        return df_boundaries

    def cluster_tumor_islets_invasive_margin(self, property_type="in_ROI_tumor_tissue", alpha=30, directory=None):
        """
        Creates a new graph only on tumor cells based on in.ROI.tumor_tissue or tissue.type
        column. Computes connected components on this new graph and determines margins.
        Creates a dataframe summarizing the distribution of phenotypes on each invasive front.
        in.ROI.tumor_tissue has 2 values True/False.
        tissue.type is in {"stroma", "tumor"}/
        :param property_type: two possible values {"in_ROI_tumor_tissue", "tissue.type"}
        :param alpha: alpha value to be used when computing concave hull
        :return: pandas dataframe with phenotypes of invasive fronts.
        """
        if directory is None:
            directory = self.working_directory
        else:
            if directory.endswith(f'GRAPH_{self.filename}/'):
                directory = directory + f'GRAPH_{self.filename}/'

        print("[" + self.filename + "] Start computing self.cluster_tumor_islets_invasive_margin()")
        if property_type == "in_ROI_tumor_tissue":
            print("[" + self.filename + "] Using cells with: in_ROI_tumor_tissue=True")
            tumor_cells, tumor_idx = list(zip(*[(value, key) for key, value in
                                                self._id_to_position_mapping.items()
                                                if self._position_to_cell_mapping[value].in_ROI_tumor_tissue]))
        else:  # property_type == "tissue.type"
            print("[" + self.filename + "]Using cells with: tissue.type=tumor")
            tumor_cells, tumor_idx = list(zip(*[(value, key) for key, value in
                                                self._id_to_position_mapping.items()
                                                if self._position_to_cell_mapping[value].tissue_type == "tumor"]))
        print("[" + self.filename + "] Computing graph for invasive fronts.")
        tumor_tissue_graph = radius_neighbors_graph(tumor_cells, self.max_dist)
        print("" + self.filename + "] Computing components for invasive fronts.")
        tumor_components = csgraph.connected_components(tumor_tissue_graph)
        print("[" + self.filename + "] Number of components_found:", tumor_components[0],
              "\n[" + self.filename + "] Largest Sizes:", Counter(tumor_components[1]).most_common(5))
        tumor_position_to_component = {}
        tumor_components_points = defaultdict(list)
        for cell_pos, number in zip(tumor_cells, tumor_components[1]):
            tumor_position_to_component[cell_pos] = number
            tumor_components_points[number].append(cell_pos)
        col_names = ["filename", "tumor_component_number", "max_dist", "alpha"] + self.all_phenotypes
        df_invasive_front = pd.DataFrame(columns=col_names)
        index = 0
        for number in tqdm.tqdm(range(tumor_components[0])):
            sequence, points = self.determine_margin_tumor(tumor_components_points[number], alpha)
            if points:
                for p in points:
                    self._position_to_invasive_margin[tuple(p)] = number
                phen = [self._position_to_cell_mapping[tuple(p)].phenotype_original
                        for p in list(set(points))]
                count_phen = Counter(phen)
                for key, value in count_phen.items():
                    df_invasive_front.loc[index, key] = value
                    df_invasive_front.loc[index, "tumor_component_number"] = number
                index += 1
        print("Successfully computed all invasive fronts.")
        # write a function to create this dataframe quicker
        df_invasive_front["filename"] = self.filename
        df_invasive_front["max_dist"] = self.max_dist
        df_invasive_front["alpha"] = alpha
        df_invasive_front = df_invasive_front.fillna(0)
        df_invasive_front.to_csv(directory + "cluster_tumor_invasive_fronts.csv", index=False, header=True)
        return df_invasive_front

    def cluster_by_component_save(self, path: str, name: str):
        """
        Save to json file the number of component for each cell in the original IF .tsv file
        (in the same order) - for Marcin IMC analysis.
        :param path: path to data with tsv files
        :param name: identifier for json
        :return: a list with number of connected component
        """
        data = pd.read_csv(path, sep="\t")
        positions = [(i, j) for i, j in zip(data["nucleus.x"], data["nucleus.y"])]
        clusters = [self._position_to_component[pos] for pos in positions]
        jsonString = json.dumps({name + "_clusters": clusters})
        jsonFile = open(path.split("#")[0] + "cluster.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()
        print("[" + self.filename + "] Clustered data saved to:", path.split("#")[0] + "cluster.json")
        return clusters

    def summarise_dataframe(self, directory):
        if directory is None:
            directory = self.working_directory
        else:
            if directory.endswith(f'GRAPH_{self.filename}/'):
                directory = directory + f'GRAPH_{self.filename}/'

        df_copy = self.initial_structure.copy()
        df_copy["component_number"] = self._connected_components[1]
        positions = list(zip(df_copy["nucleus.x"], df_copy["nucleus.y"]))
        df_copy["immune_neighbour"] = [self._immune_position_to_component[pos]
                                       for pos in positions]
        df_copy["component_margin"] = [self._position_to_if_margin[pos]
                                       for pos in positions]
        df_copy["invasive_front"] = [self._position_to_invasive_margin[pos]
                                     for pos in positions]
        df_copy["is_immune"] = [False if self._position_to_cell_mapping[pos].marker_is_active(Marker[self.panel].CK)
                                else True for pos in positions]
        df_copy.to_csv(directory + "df_summary_" + str(self.filename) + ".csv")
        del df_copy
        gc.collect()

    def characterize_CK_islets(self):
        """Returns dataframe with one row, each column corresponds to the component number,
        value in 0-th row to the number of cells in each component."""
        ctr = Counter(self.connected_components()[1])
        ctr = {k: [v] for k, v in ctr.items()}
        return pd.DataFrame.from_dict(ctr)

    def select_CK_component_IDs(self, number):
        """Returns a list of cell's ids from the selected component."""
        return np.where(self.connected_components()[1] == number)[0]

    def get_CK_component_by_number(self, number):
        """"Returns a list of cell's positions from the selected component."""
        return self._connected_components_positions[number]

    def determine_margin(self, number, alpha=20):
        points_for_concave_hull = self.select_CK_component_points(number)
        if len(points_for_concave_hull) <= 4:
            return None, None

        concave_hull, edge_points = [], []
        try:
            concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
        except:
            # alpha_shape failes - may be due to colinear points return empty lists
            print("[" + self.filename + "] Empty edges for component: {}", number)

        if not edge_points:
            return [], [], defaultdict(list)
        elif isinstance(concave_hull, shapely.geometry.LineString):
            result = Graph_additional._determine_margin_helper(boundary=concave_hull.boundary, number=number)
            return result
        else:
            _inv_mar_seq_tmp, _inv_mar_tmp = [], []
            _position_to_component = defaultdict(list)
            for geom in concave_hull.geoms:
                margin_edge_sequence, margin_positions, position_to_component = Graph_additional._determine_margin_helper(
                    boundary=geom.boundary, number=number)
                _inv_mar_seq_tmp.extend(margin_edge_sequence)
                _inv_mar_tmp.extend(margin_positions)
                _position_to_component.update(position_to_component)
            return _inv_mar_seq_tmp, _inv_mar_tmp, _position_to_component

    def determine_margin_tumor(self, points, alpha=20):
        number = -2
        points_for_concave_hull = points
        if len(points_for_concave_hull) <= 4: return None, None

        # concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
        concave_hull, edge_points = [], []
        try:
            concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
        except:
            print("[" + self.filename + "] WARN: Margin cannot be determined, possibly cells are collinear.", )
            return [], []
        # Margin is not determined return empty lists.
        if not edge_points:
            return [], []
        elif isinstance(concave_hull, shapely.geometry.LineString):
            result = Graph_additional._determine_margin_helper(boundary=concave_hull.boundary, number=number)[:2]
            return result
        elif isinstance(concave_hull, shapely.geometry.Polygon):
            result = Graph_additional._determine_margin_helper(boundary=concave_hull.boundary, number=number)[:2]
            return result
        else:
            _inv_mar_seq_tmp, _inv_mar_tmp = [], []
            for geom in concave_hull.geoms:
                margin_edge_sequence, margin_positions = Graph_additional._determine_margin_helper(
                    boundary=geom.boundary, number=number)[:2]
                _inv_mar_seq_tmp.extend(margin_edge_sequence)
                _inv_mar_tmp.extend(margin_positions)

            return _inv_mar_seq_tmp, _inv_mar_tmp

    def determine_all_margins(self, alpha=20, min_islet_size=10, run_parallel=True):
        """Determine margins of all islets with more than min_islet_size cells inside."""
        start_time = time.process_time()
        mid_time = start_time
        islet_sizes = self.characterize_CK_islets()
        if run_parallel:
            start_time = time.process_time()
            if not self._connected_components:
                self.connected_components()
            print("Preparing data with components for parallelization.")
            selected_islets = [(idx, islet_sizes.at[0, idx], self.get_CK_component_by_number(idx))
                               for idx in islet_sizes
                               if islet_sizes.at[0, idx] >= min_islet_size and idx != -1]
            selected_islets = [(idx, component) for idx, size, component
                               in sorted(selected_islets, key=lambda x: x[1], reverse=True)]
            print("Runtime, data prepared: {}. ".format(round(time.process_time() - start_time, 3)))
            number_of_cpu = int(0.75 * multiprocessing.cpu_count())
            number_of_islets = len(selected_islets)
            n = int(number_of_islets / number_of_cpu) + 1
            if n <= 2:
                n = int(number_of_islets / 10) + 1
            else:
                n = number_of_cpu
            islet_subsets = list(Graph_additional.divide_chunks(selected_islets, n, sorted_list=True))
            print("[MULTIPROCESSING] Calculate {} margins.\nTrying to invoke {} tasks on {} cpu".format(
                number_of_islets, n, number_of_cpu
            ))
            with multiprocessing.Pool(processes=32, maxtasksperchild=32) as pool:
                params = [(islets, alpha) for islets in islet_subsets]
                ## MAP ASYNC
                results = pool.starmap_async(Graph_additional._parallel_helper_margin_detection, params)
                for p in results.get():
                    inv_mar_seq, inv_mar, position_to_component = p
                    self._invasive_margins_sequence.update(inv_mar_seq)
                    self._invasive_margins.update(inv_mar)
                    self._position_to_component.update(position_to_component)
        else:
            selected_islets = [(idx, islet_sizes.at[0, idx])
                               for idx in islet_sizes
                               if islet_sizes.at[0, idx] >= min_islet_size and idx != -1]
            selected_islets = [(idx, size)
                               for idx, size in sorted(selected_islets, key=lambda x: x[1], reverse=True)]
            # WITHOUT MULTIPROCESSING
            for islet_number, islet_size in selected_islets:
                print("Determine margin for islet number: " + str(islet_number) + " of size " + str(
                    islet_size))
                inv_mar_seq, inv_mar, position_to_component = self.determine_margin(islet_number, alpha=alpha)
                self._invasive_margins_sequence[islet_number] = inv_mar_seq
                self._invasive_margins[islet_number] = inv_mar
                self._position_to_component.update(position_to_component)
                print("Runtime, determine_margin(): {}. ".format(round(time.process_time() - mid_time, 3)))
                mid_time = time.process_time()
        print(("Runtime, determine_all_margins(): {}. " + "Number of margins analyzed: {}").format(
            round(time.process_time() - start_time, 3), len(selected_islets)))

    def get_invasive_margin(self, number):
        """Returns a set of cells belonging to the invasive margins from the selected component."""
        # in order to compute that we need to run self.determine_all_margins first
        if len(self._invasive_margins) == 0:
            self.determine_all_margins()
        cells_positions = self._invasive_margins[number]
        invasive_margin = []
        for i in cells_positions:
            cell_i = self._position_to_cell_mapping[i]
            invasive_margin.append(cell_i)
            for j in self._graph_dict[cell_i.id, :].indices:
                invasive_margin.append(self._position_to_cell_mapping[self._id_to_position_mapping[j]])
        return set(invasive_margin)

    def characterize_invasive_margin(self, number, plot_labels=False, save_plot=True,
                                     display_plots=True):
        directory = self.working_directory + "/Components/component_" + str(number)
        all_phenotypes = []
        inv_mar = [cell.phenotype_label for cell in self.get_invasive_margin(number)]
        all_phenotypes.extend(inv_mar)
        margin_number = [number] * len(inv_mar)
        data = {'Margin Number': margin_number, 'Cell Type': all_phenotypes}
        pal = Graph_additional._create_color_dictionary(set(all_phenotypes))
        df = pd.DataFrame(data)
        ax = pd.crosstab(df['Margin Number'], df['Cell Type']).apply(lambda r: r / r.sum() * 100, axis=1)
        tmp = pd.crosstab(df['Margin Number'], df['Cell Type'])
        tmp.to_csv(directory + "/-description-margin-{}.csv".format(self.max_dist))
        if save_plot:
            total_cells = tmp.transpose().sum().tolist()
            ax_1 = ax.plot.bar(figsize=(16, 9), stacked=True, rot=0, color=pal)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       fancybox=True, shadow=True, ncol=7, title="Gene Marker")
            plt.xlabel('Margin Number')
            plt.ylabel('Percent Distribution')
            for idx, rec in enumerate(ax_1.patches):
                height = rec.get_height()
                _row = idx % tmp.shape[0]
                _col = idx // tmp.shape[0]
                if plot_labels:
                    ax_1.text(rec.get_x() + rec.get_width() / 2,
                              rec.get_y() + height / 2,
                              "{:.0f}%".format(height) if height >= 5 else " ",
                              ha='center',
                              va='bottom',
                              rotation=90)
            labels = [item.get_text() + " (" + str(total_cells[idx]) + ")" for idx, item in
                      enumerate(ax_1.get_xticklabels())]
            ax_1.set_xticklabels(labels)
            plt.xticks(rotation=90)
            plt.savefig(directory + "/-description-margin-{}.pdf".format(self.max_dist))
            if display_plots:
                plt.show()

    def summarize_column(self, column, include_neighbors=True, neighbors_dist=None, save=False,
                         directory=None, name='component'):
        if directory is None: directory = self.working_directory
        all_phenotypes = list(self.initial_structure['phenotype'].unique())

        columns = [f"{name}.number", f"{name}.size"] + all_phenotypes
        if 'in.ROI.tumor_tissue' in self.initial_structure.columns:
            columns.append("in.ROI.tumor.%")
        summary = dict([(key, []) for key in columns])
        for cn in self.initial_structure[column].unique():
            print(cn)
            cells = self.initial_structure.loc[self.initial_structure[column] == cn]
            if len(cells) == 0:
                continue
            if include_neighbors:
                if neighbors_dist is None: neighbors_dist = self.max_dist
                inds, D = self.get_cells_neighbors(cells_selected=cells,
                                                        max_dist=neighbors_dist)
                neighbors = self.initial_structure.loc[self.initial_structure['index'].isin(inds)]
                cells = pd.concat([cells, neighbors])
                cells.drop_duplicates('index', inplace=True)

            ph_counts = dict(cells['phenotype'].value_counts())
            for ph in all_phenotypes:
                if ph in ph_counts.keys():
                    summary[ph].append(ph_counts[ph])
                else:
                    summary[ph].append(0)
            summary[f"{name}.number"].append(cn)
            summary[f"{name}.size"].append(len(cells))
            if "in.ROI.tumor.%" in columns:
                summary["in.ROI.tumor.%"].append(
                    len(cells.loc[cells['in.ROI.tumor_tissue'] == True]) / len(cells))

        summary = pd.DataFrame.from_dict(summary)
        if save:
            if include_neighbors:
                text = f'included_{neighbors_dist}_µm_neighbors'
            else:
                text = 'no_neighbors'
            summary.to_csv(f'{directory}/summarized_{column}_{text}_{self.filename}.tsv', sep='\t')

        return summary
