import time
import logging
import pylab as plt
import numpy as np

from matplotlib import collections as mc
from scipy.spatial import Delaunay
from scipy.spatial import distance
from matplotlib.lines import Line2D

import tumor_islets.graph.Graph as Graph
import tumor_islets.graph.graph_additional as graph_additional
from tumor_islets.graph.Cell import Cell, Marker

logging.basicConfig(level=logging.INFO)


class PlotTME(object):
    def __int__(self, figsize=(16, 12), subplots=None):
        """Specify general plot options for this figure"""
        plt.rcParams['figure.figsize'] = figsize

    @staticmethod
    def get_margin_border(ax, lines, graph, componentNumber, plot_labels):
        "Get margin border for given component"
        mid_time = time.process_time()
        components_list = []
        if type(componentNumber).__name__ == "int":
            components_list = [componentNumber]
        elif componentNumber is None:
            components_list = graph._invasive_margins_sequence.keys()
        for cmpNum in components_list:
            _inv_mar_seq = graph._invasive_margins_sequence[cmpNum]
            lines["Margin-Border"].extend(_inv_mar_seq)
            if _inv_mar_seq and plot_labels:
                (lab_x1, lab_y1), (lab_x2, lab_y2) = _inv_mar_seq[0]
                ax.text((lab_x1 + lab_x2) / 2, (lab_y1 + lab_y2) / 2,
                        cmpNum, fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="pink", ec="cyan", lw=1, alpha=0.5),
                        ha='center', va='center')
        print("Runtime, Graph.plot():pMarginBorder: {}.".format(round(time.process_time() - mid_time, 3)))

    @staticmethod
    def get_islet_edges(graph, subset, lines, other_cells_x, other_cells_y, other_cells_phenotype):
        mid_time = time.process_time()
        # Lines (edges) connecting cells within CK-islet and neighbouring not CK-cells
        for i in subset:
            cell_i = graph._position_to_cell_mapping[graph._id_to_position_mapping[i]]
            for j in graph._graph_dict[i, :].indices:
                cell_j = graph._position_to_cell_mapping[graph._id_to_position_mapping[j]]
                line = (cell_i.position,
                        cell_j.position)
                if j in subset:
                    lines["Islet-Islet"].append(line)
                else:
                    other_cells_x.append(cell_j.x)
                    other_cells_y.append(cell_j.y)
                    other_cells_phenotype.append(cell_j.phenotype_label)
                    lines["Margin-Islet"].append(line)
        lines["Islet-Islet"] = [(x, y) for x, y in set(lines["Islet-Islet"])]
        lines["Margin-Islet"] = [(x, y) for x, y in set(lines["Margin-Islet"])]
        other_cells_x = list(np.array(other_cells_x))
        other_cells_y = list(np.array(other_cells_y))
        other_cells_phenotype = list(np.array(other_cells_phenotype))
        print("Runtime, Graph.plot():pIsletEdges or pOuterEdges: {}.".format(
            round(time.process_time() - mid_time, 3)))
        return other_cells_x, other_cells_y, other_cells_phenotype, lines

    @staticmethod
    def get_outer_cells(graph, componentNumber, other_cells_x, other_cells_y, other_cells_phenotype):
        mid_time = time.process_time()
        cx = graph._graph_dict.tocoo()
        for i, j, _ in zip(cx.row, cx.col, cx.data):
            cell_j = graph._position_to_cell_mapping[graph._id_to_position_mapping[j]]
            cell_i = graph._position_to_cell_mapping[graph._id_to_position_mapping[i]]
            if cell_i.position in graph._position_to_component and \
                    graph._position_to_component[cell_i.position] == componentNumber and \
                    not cell_j.marker_is_active(Marker.CK):  # activities["CK"]:
                other_cells_x.append(cell_j.x)
                other_cells_y.append(cell_j.y)
                other_cells_phenotype.append(cell_j.phenotype_label)
        other_cells_x = np.array(other_cells_x)
        other_cells_y = np.array(other_cells_y)
        other_cells_phenotype = np.array(other_cells_phenotype)
        print("Runtime, Graph.plot():pOuterCells: {}.".format(
            round(time.process_time() - mid_time, 3)))
        return other_cells_x, other_cells_y, other_cells_phenotype

    @staticmethod
    def connect_margin_edges(graph, componentNumber, lines):
        """get boarder of given cellular islet or invasive front if plotting whole graph"""
        mid_time = time.process_time()
        # Lines (edges) connecting cells within the margin limits (< 50um from the islet border)
        components_list = []
        if type(componentNumber).__name__ == "int":
            components_list = [componentNumber]
        elif componentNumber is None:
            components_list = graph._invasive_margins_sequence.keys()
        for id_ in components_list:
            margin_cells = graph.get_invasive_margin(id_)
            if margin_cells:
                _polygon_points = [cell.position for cell in margin_cells]
                tri = Delaunay(_polygon_points)
                thresh = graph.max_dist
                short_edges = set()
                dev_null = set()
                new_tri = []
                for tr in tri.vertices:
                    segment_count = 0
                    for i in range(3):
                        edge_idx0 = tr[i]
                        edge_idx1 = tr[(i + 1) % 3]
                        p0 = _polygon_points[edge_idx0]
                        p1 = _polygon_points[edge_idx1]
                        if distance.euclidean(p1, p0) <= thresh:
                            segment_count += 1
                            if segment_count > 1:
                                new_tri.append(tr)
                                break
                        if (edge_idx1, edge_idx0) in dev_null or (edge_idx1, edge_idx0) in short_edges:
                            continue
                        if distance.euclidean(p1, p0) <= thresh:
                            short_edges.add((edge_idx0, edge_idx1))
                        else:
                            dev_null.add((edge_idx0, edge_idx1))
                lines["Margin"] = [[_polygon_points[i], _polygon_points[j]] for i, j in short_edges]
                tri_x = np.array([x for x, y in _polygon_points])
                tri_y = np.array([y for x, y in _polygon_points])
                pts = np.zeros((len(_polygon_points), 2))
                pts[:, 0] = tri_x
                pts[:, 1] = tri_y
                plt.tripcolor(pts[:, 0], pts[:, 1], np.array(len(new_tri) * [0.2]),
                              triangles=new_tri, alpha=0.2, edgecolors='k')
                del dev_null
        print("Runtime, Graph.plot():pMarginEdges: {}.".format(round(time.process_time() - mid_time, 3)))
        return lines

    @staticmethod
    def add_lines(lines, ax, marginAlpha, isletAlpha, marginIsletAlpha, pIsletEdges, pOuterEdges,
                  pMarginBorder, pMarginEdges):
        mid_time = time.process_time()
        lines_properties_dict = {"Islet-Islet": {"color": "Red", "alpha": isletAlpha,
                                                 "linestyle": (0, (5, 10)), "draw": pIsletEdges},
                                 "Margin-Islet": {"color": "Green", "alpha": marginIsletAlpha,
                                                  "linestyle": (0, (3, 5, 1, 5)), "draw": pOuterEdges},
                                 "Margin-Border": {"color": "Purple", "alpha": marginAlpha,
                                                   "linestyle": 'solid', "draw": pMarginBorder},
                                 "Margin": {"color": "Purple", "alpha": marginAlpha,
                                            "linestyle": (0, (5, 10)), "draw": pMarginEdges}}
        for interaction in lines.keys():
            if lines_properties_dict[interaction]["draw"]:
                line_collection = mc.LineCollection(lines[interaction],
                                                    linestyle=lines_properties_dict[interaction]["linestyle"],
                                                    colors=lines_properties_dict[interaction]["color"],
                                                    label=interaction,
                                                    linewidth=0.5,
                                                    alpha=lines_properties_dict[interaction]["alpha"])
                ax.add_collection(line_collection)
                ax.autoscale()
                ax.margins(0.1)
                ax.grid(True)
        print("Runtime, Graph.plot():for interaction in lines.keys():: {}.".format(
            round(time.process_time() - mid_time, 3)))
        return lines_properties_dict

    @staticmethod
    def make_proxy(color, marker, label, **kwargs):
        if marker == 'o':
            return Line2D([0], [0], color='w', markerfacecolor=color, marker=marker, label=label, **kwargs)
        else:
            return Line2D([0], [0], color=color, label=label, **kwargs)

    def color_phenotypes(self, ax, xs, ys, phenotypes, cells_to_plot,
                         pOuterCells, other_cells_x, other_cells_y, other_cells_phenotype,
                         isletAlpha, marginIsletAlpha):
        mid_time = time.process_time()
        set_of_all_phenotypes = set(phenotypes).union(set(other_cells_phenotype))
        color_dict_phenotypes = graph_additional._create_color_dictionary(set_of_all_phenotypes)

        for phenotype in set(phenotypes):
            to_plot = np.where((phenotypes == phenotype))[0]
            ax.scatter(xs[to_plot], ys[to_plot], alpha=isletAlpha,
                       color=color_dict_phenotypes[phenotype],
                       label=phenotype, marker=".", s=5)  # s=s
        logging.debug(("Points collections: \n" + "Islet: {};").format(len(cells_to_plot)))
        if pOuterCells:
            for phenotype in set(other_cells_phenotype):
                to_plot = np.where(other_cells_phenotype == phenotype)[0]
                ax.scatter(other_cells_x[to_plot], other_cells_y[to_plot], color=color_dict_phenotypes[phenotype],
                           alpha=marginIsletAlpha, label=phenotype, s=5)  # s=s size of point
        logging.debug(("Points collections: \n" + "Outer (nonCK): {};").format(len(other_cells_x)))

        proxies = [self.make_proxy(color, marker='o', label=name, markersize=5) for name, color in
                   color_dict_phenotypes.items()]  # markersize=s
        second_legend = ax.legend(handles=proxies, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                                  fancybox=True, shadow=True, ncol=7, title="Gene Marker")
        ax.add_artist(second_legend)
        print("Runtime, Graph.plot():pVert: {}.".format(round(time.process_time() - mid_time, 3)))

    def plot(self, graph: Subgraph, subset=None, componentNumber=None, pMarginBorder=True, pVert=True,
             pIsletEdges=False, pMarginEdges=True, pOuterEdges=False, pOuterCells=True,
             plot_labels=False, isletAlpha=0.075, marginIsletAlpha=0.5, marginAlpha=0.075,
             display_plots=True, s=5, figsize=(16, 12), path_to_save="tmp", verbose=False):

        """Plot function for subgraphs, seperate compartments or whole graphs:
        :param subset -
        :param componentNumber: id number of cellular islet to plot
        :param pMarginBorder:
        :param pVert:
        :param pIsletEdges: plot margin edges
        :param pMarginEdges:
        :param pOuterEdges:
        :param pOuterCells:
        :param plot_labels:
        :param isletAlpha:
        :param marginIsletAlpha:
        :param marginAlpha:
        :param display_plots: bool
        :param s: size of cell on plot"""
        start_time = time.process_time()
        mid_time = start_time
        print("Graph.plot(): START")
        plt.rcParams['figure.figsize'] = figsize
        fig, ax = plt.subplots()
        # select cells
        if not subset and not componentNumber:
            subset = graph._id_to_position_mapping.keys()
        if componentNumber is not None:
            subset = graph.select_neighborhood_IDs(componentNumber).tolist()

        cells_to_plot = list(map(graph._position_to_cell_mapping.get, list(map(graph._id_to_position_mapping.get,
                                                                               subset))))
        xs = np.array([cell.x for cell in cells_to_plot])
        ys = np.array([cell.y for cell in cells_to_plot])
        phenotypes = np.array([cell.phenotype_label for cell in cells_to_plot])
        lines = {"Islet-Islet": [], "Margin-Islet": [], "Margin-Border": [], "Margin": []}
        other_cells_phenotype, other_cells_x, other_cells_y = [], [], []
        print("Runtime, Graph.plot():compute cells_to_plot: {}.".format(round(time.process_time() - mid_time, 3)))
        # if verbose:
        #     for cell in cells_to_plot:
        #         print(cell)

        # Gathering lines
        # Lines (edges) describing a path of the islet border
        if pMarginBorder:
            self.get_margin_border(ax, lines, graph, componentNumber, plot_labels)

        if pIsletEdges or pOuterEdges:
            other_cells_x, other_cells_y, other_cells_phenotype, lines = self.get_islet_edges(graph, subset, lines,
                                                                                              other_cells_x,
                                                                                              other_cells_y,
                                                                                              other_cells_phenotype, )

        if pOuterCells and (componentNumber is not None):
            other_cells_x, other_cells_y, other_cells_phenotype = self.get_outer_cells(graph, componentNumber,
                                                                                       other_cells_x, other_cells_y,
                                                                                       other_cells_phenotype, )

        # Lines (edges) conecting cells within the margin limits (< 50um from the islet border)
        if pMarginEdges:
            self.connect_margin_edges(graph, componentNumber, lines)

        lines_properties_dict = self.add_lines(lines, ax, marginAlpha, isletAlpha, marginIsletAlpha,
                                               pIsletEdges, pOuterEdges, pMarginBorder, pMarginEdges)

        proxies = [self.make_proxy(properties["color"], marker=None, label=name, alpha=properties["alpha"],
                                   linestyle=properties["linestyle"]) for name, properties in
                   lines_properties_dict.items()]

        first_legend = ax.legend(handles=proxies, loc='upper right')
        ax.add_artist(first_legend)
        # Gathering points
        if pVert:
            self.color_phenotypes(ax, xs, ys, phenotypes, cells_to_plot,
                                  pOuterCells, other_cells_x, other_cells_y, other_cells_phenotype,
                                  isletAlpha, marginIsletAlpha)

        ax.autoscale()
        plt.savefig(path_to_save + "-margin-{}.pdf".format(graph.max_dist))
        if display_plots:
            plt.show()
        print(("Runtime, plot(): ({}).\n" +
               "EDGES -- Islet-Islet: {}; Margin-Margin: {}; Margin-Islet: {}, Margin Border: {}\n" +
               "VERTICES -- Inner cells: {}; Outer cells: {}").format(
            round(time.process_time() - start_time, 3),
            len(lines["Islet-Islet"]), len(lines["Margin"]),
            len(lines["Margin-Islet"]), len(lines["Margin-Border"]),
            len(xs), len(other_cells_x)))
