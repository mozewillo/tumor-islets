import json
import pandas as pd
import time
from collections import defaultdict
from tumor_islets.graph import graph_additional
import scipy
from matplotlib import collections as mc
import numpy as np
from matplotlib import pyplot as plt, cm
from shapely.geometry import MultiLineString
from shapely.ops import polygonize, unary_union
from descartes import PolygonPatch

from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csgraph


def plot_layers(sequences, n=None, legend=False, figsize=(16, 9)):
    """
    The function to visualize n first layers of a tumor islet as lines (without background color).
    Args:
        sequences: a list of lists (subsequent layers)
        n: number of layers to plot
        legend: boolean, if include legend (for large n, the legend is deprecated)
        figsize: figure size in matplotlib, by default equal to (16, 9)

    Returns:
        a matplotlib figure visualizing first n layers from a sequence object
    """
    if not n:
        n = len(sequences)
    if n > 30 and legend:
        print("[WARNING] legend is deprecated.")

    fig, ax = plt.subplots(figsize=figsize)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    i = 0
    for sequence in sequences:
        if legend:
            collection_lines = mc.LineCollection(sequence, linestyle="-",
                                                 label="Layer " + str(i), colors=next(color))
            i += 1
        else:
            collection_lines = mc.LineCollection(sequence, linestyle="-", colors=next(color))
        ax.add_collection(collection_lines)
    ax.set_aspect("equal")
    plt.plot()
    plt.grid(True)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


def plot_layers_polygons(sequences, n=None, legend=False, figsize=(16, 9)):
    """
    This function visualizes n first selected layers of a tumor islet (if n is not defined,
    then it selects first 30 layres) as polygons (lines with background coloring).

    Args:
        sequences: a list of lists (subsequent layers)
        n: number of layers to plot
        legend: boolean, if include legend (for large n, the legend is deprecated)
        figsize: figure size in matplotlib, by default equal to (16, 9)

    Returns:
        a matplotlib figure visualizing n first layers as polygons.
    """
    if not n:
        n = len(sequences)
    if n > 30 and legend:
        print("[WARNING] legend is deprecated.")
    fig, ax = plt.subplots(figsize=figsize)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))

    for sequence in sequences:
        c = next(color)
        multi = MultiLineString(sequence)
        polygons = list(polygonize(unary_union(multi)))
        for polygon in polygons:
            patch = PolygonPatch(polygon, facecolor=c, edgecolor=c, alpha=0.3)
            ax.add_patch(patch)

    ax.set_aspect("equal")
    plt.plot()
    return fig


def prepare_slicing_dataframe(
        points,
        filename,
        component_number,
):
    """
    This function prepares a dataframe to store the results of slicing algorithm. It stores
    the positions and phenotypes of cells from a given tumor islet, adds an additional column
    checking if a cell is immune or not. It creates empty columns for slicing results such as
    steep_region_number, steepness, on_margin and layer_number.

    Args:
        points: a dictionary with two keys: positions and phenotypes, and lists as values
        filename: a name of an analysed file, string
        component_number: the number of an analysed tumor islet

    Returns:
        a pandas dataframe for storing the results of slicing algorithm.
    """
    df = pd.DataFrame(columns=["filename", "component_number", "nucleus.x", "nucleus.y",
                               "phenotype", "is_immune", "on_margin", "layer_number",
                               "steep_region_number", "steepness"])
    df["nucleus.x"] = [p[0] for p in points["positions"]]
    df["nucleus.y"] = [p[1] for p in points["positions"]]
    df["filename"] = filename
    df["component_number"] = component_number
    df["phenotype"] = points["phenotypes"]
    df["is_immune"] = df["phenotype"].apply(lambda x: True if "CK+" not in x else False)
    return df


def compute_margin_sequences(
    CK_cells, 
    alpha=100,
):
    """
    The key function for computing the sequence of layers based on a list of CK cells positions.
    It detects the alpha shape of a set of points using the helper_margin_slices function,
    until there are less than 6 cells left inside a component.

    Args:
        CK_cells: a list of cell positions in a form [(x1, y1), ...., (xn, yn)]
        alpha: the value of alpha for the alpha-shape algorithm

    Returns:
        a tuple with margin_sequences, position_to_margin and margin_layers.
        The margin_sequences is a list of lists of all detected layers in a sequence form.
        Position_to_margin is a dictionary with cell positions as keys and layer number as values.
        Margin_layers is a dictionary mapping a layer number to a set of cells that form this layer.
    """
    margin, margin_sequences = set(), []
    margin_layers, position_to_margin = {}, defaultdict(lambda: -1)
    layer_number, zero_margin = 0, 0

    while len(CK_cells - margin) >= 5 and zero_margin != 1:
        CK_cells = CK_cells - margin
        sequence, margin = graph_additional.helper_margin_slices(CK_cells, alpha=alpha)
        margin_sequences.append(sequence)
        margin = set(margin)
        margin_layers[layer_number] = margin
        if sequence:
            for point in margin:
                position_to_margin[point] = layer_number
        if len(margin) == 0:
            zero_margin = 1
        layer_number += 1

    return margin_sequences, position_to_margin, margin_layers


def assign_steepness(
        pos_to_mar,
        new_dist=50,
        min_steepness=5,
):
    """
    This function calculates the steepness attribute of a cell (maximum difference between the
    layer number of a given cell A and all its neighbouring cells within a distance of new dist),
    then it selects steep cells based on min_steepness parameter and their neighbours.

    Args:
        pos_to_mar: a dictionary mapping a cell position to a layer number
        new_dist: the radius for a radius neighbourhood graph created on steep cells
        min_steepness: the minimum value of a steepness attribute of a cell to be considered
        as steep

    Returns:
        A tuple with pos_to_steepness - a dictionary mapping cell positions to steepness value,
        and a list of all steep cells and their neighbours.
    """
    points_margin = [k for k, v in pos_to_mar.items() if v >= 0]
    pos_to_steepness = defaultdict(int)
    steep_points, steep_neighbors = [], []

    ctree = scipy.spatial.cKDTree(points_margin)
    for point in points_margin:
        neighbors = ctree.query_ball_point(np.array(point), new_dist)
        max_steep = 0
        for neighbor_ind in neighbors:
            neighbor = points_margin[neighbor_ind]
            margin_diff = abs(pos_to_mar[point] - pos_to_mar[neighbor])
            if margin_diff > max_steep:
                max_steep = margin_diff
        pos_to_steepness[point] = max_steep
        if max_steep >= min_steepness:
            steep_neighbors.extend([points_margin[ind] for ind in neighbors])

    return pos_to_steepness, list(set(steep_neighbors))


def detect_steep_regions(
    steep_plus_neighbors, 
    new_dist=50,
):
    """
    This function detects steep regions (connected components build on top of a radius
    neighbourhood graph with radius = new_dist) using previously selected steep cells
    along with their neighbours, using the connected_components function.

    Args:
        steep_plus_neighbors: a list of positions of steep cells and their neighbours
        new_dist: the radius for a radius neighbourhood graph created on steep cells

    Returns:
        A dictionary mapping a cell position to a steep region number (and -1 if a cell
        is not inside any steep region)
    """
    graph_steep = radius_neighbors_graph(steep_plus_neighbors, new_dist,
                                         mode="connectivity", include_self=False)
    components_steep = csgraph.connected_components(graph_steep)
    pos_to_steep_region = defaultdict(lambda: -1)
    for i in range(components_steep[0]):
        points = list(np.array(steep_plus_neighbors)[np.where(components_steep[1] == i)])
        for point in points:
            pos_to_steep_region[tuple(point)] = i
    return pos_to_steep_region


def assign_immune_cells(
        df,
        pos_to_mar,
        pos_to_steep,
        pos_to_steep_region,
):
    """
    This function assigns attributes to immune cells using cKD queries. For each immune cell
    it finds its nearest neighbour from CK positive cells and assigns the attributes of this
    nearest CK neighbour to the given immune cell.

    Args:
        df: a dataframe prepared to store slicing results
        pos_to_mar: a dictionary mapping cell positions to layer number
        pos_to_steep: a dictionary mapping cell positions to steepness value
        pos_to_steep_region: a dictionary mapping cell positions to steep region number

    Returns:
        a pandas dataframe with assigned attributes such as layer number, steepness, steep_region_number
        and on_margin.
    """
    CK_points = [k for k, v in pos_to_mar.items() if v >= 0]
    df_immune = df[df["is_immune"] == True]
    ctree = scipy.spatial.cKDTree(CK_points)

    for ind in df_immune.index:
        point = (df_immune.loc[ind, "nucleus.x"], df_immune.loc[ind, "nucleus.y"])
        ds, ind = ctree.query(np.array(point), 1)
        CK_neighbor = CK_points[ind]
        # update dictionaries
        pos_to_mar[point] = pos_to_mar[CK_neighbor]
        pos_to_steep[point] = pos_to_steep[CK_neighbor]
        pos_to_steep_region[point] = pos_to_steep_region[CK_neighbor]

    df["layer_number"] = [pos_to_mar[(x, y)] for x, y in zip(df["nucleus.x"], df["nucleus.y"])]
    df["steepness"] = [pos_to_steep[(x, y)] for x, y in zip(df["nucleus.x"], df["nucleus.y"])]
    df["steep_region_number"] = [pos_to_steep_region[(x, y)] for x, y in zip(df["nucleus.x"], df["nucleus.y"])]
    df["on_margin"] = df["layer_number"].apply(lambda x: True if x == 0 else False)
    return df


def slice_tumor_islet(
        points_path,
        filename,
        component_number,
        output_path=None,
        alpha=100,
        min_steepness=5,
        new_dist=50,
        plot=True,
):
    """
    The final function to perform slicing of a tumor islet. It first prepares a dataframe
    for slicing outputs, then computes the sequence of margins using the alpha-shape algorithm.
    In the next steps, it calculates steepness of cells, detects steep regions and assigns
    immune cells to layers and regions.

    Args:
        points_path: a path to a .json file with cell positions and phenotypes
        filename: the name of the analysed file
        component_number: the number of an analysed tumor islet
        output_path: a prefix of the output path
        alpha: the value of alpha for alpha-shape algorithm
        min_steepness: the minimum value of steepness for a cell to be considered as a steep cell
        new_dist: the value of radius for radius neighbourhood graph construction
        plot: boolean, if True it creates additional visualizations during the execution

    Returns:
        A tuple with df_slicing, sequences and margin_slices. The df_slicing is a pandas
        dataframe that where each row represents a cell, and columns stores all important
        features computed by slicing algorithm. Sequences is a list of lists with layers
        detected by the alpha-shape algorithm stored in a sequence form. Margin_slices is
        a dictionary that maps a layer number to a set of cells that are assigned to this
        specific layer.
    """
    points = json.load(open(points_path))

    df_slicing = prepare_slicing_dataframe(points, filename, component_number)
    CK_cells_df = df_slicing[df_slicing["is_immune"] == False]
    CK_cells = set([(x, y) for x, y in zip(CK_cells_df["nucleus.x"], CK_cells_df["nucleus.y"])])
    print(f"Number of CK cells in islet = {len(CK_cells)}")

    if len(CK_cells) < 50:
        print(f"[WARNING] Less than 50 CK+ cells in an islet, slicing not performed.")
        return 0

    start_time = time.perf_counter()
    sequences, pos_to_mar, margin_slices = compute_margin_sequences(CK_cells, alpha=alpha)
    print(f"[COMPUTE_MARGIN_SEQUENCES (concave hull)] time: {time.perf_counter() - start_time}")

    if plot:
        fig = plot_layers(sequences, legend=False)
        fig.show()
        fig_poly = plot_layers_polygons(sequences, legend=False)
        fig_poly.show()

    start_time = time.perf_counter()
    pos_to_steep, steep_neighbors = assign_steepness(pos_to_mar, new_dist, min_steepness)
    print(f"[POINTS_STEEPNESS] time: {time.perf_counter() - start_time}")

    steep_points = [k for k, v in pos_to_steep.items() if v >= min_steepness]
    if not steep_points:
        print("[WARNING] No steep points found, try smaller min_steepness parameter.")
        pd.DataFrame({}).to_csv(output_path + "_sliced_df.csv") # For snakemake errors
        return 0

    start_time = time.perf_counter()
    pos_to_steep_region = detect_steep_regions(steep_points + steep_neighbors, new_dist)
    print(f"[DETECT STEEP REGIONS] time: {time.perf_counter() - start_time}")

    df_slicing = assign_immune_cells(df_slicing, pos_to_mar, pos_to_steep, pos_to_steep_region)

    if output_path:
        results_dict = {"sequences": sequences,
                        "margin_slices": {k: list(v) for k, v in margin_slices.items()}}
        results_dict.update(points)
        df_slicing.to_csv(output_path + "_sliced_df.csv", index=False)
        with open(output_path + '_slicing_points.json', "w") as outfile:
            json.dump(results_dict, outfile)

    return df_slicing, sequences, margin_slices
