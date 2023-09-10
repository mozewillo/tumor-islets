import pandas as pd
import scipy
import scipy.spatial
import time
import plotly.express as px
import json
from matplotlib import cm
from matplotlib import collections as mc
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def prepare_dataframe(points, pos_to_mar, filename, component_number):
    """
    Prepares a dataframe for cutting algorithm.
    @param points: dictionary with two keys: positions and phenotypes of points inside a component
    @param pos_to_mar: dictionary: {tuple(x, y): margin_number}
    @param filename: patient/sample id
    @param component_number: component_number
    @return: dataframe with positions, phenotype, is_immune, layer_number and cut number
    """
    cuts_df = pd.DataFrame(columns=["filename", "component_number", "nucleus.x", "nucleus.y",
                                    "phenotype", "is_immune", "layer_number", "cut_number"])
    cuts_df["nucleus.x"] = [p[0] for p in points["positions"]]
    cuts_df["nucleus.y"] = [p[1] for p in points["positions"]]
    cuts_df["phenotype"] = points["phenotypes"]
    cuts_df["is_immune"] = cuts_df["phenotype"].apply(lambda x: True if "CK+" not in x else False)
    cuts_df["layer_number"] = [pos_to_mar[(x, y)] for x, y in zip(cuts_df["nucleus.x"], cuts_df["nucleus.y"])]
    cuts_df["filename"] = filename
    cuts_df["component_number"] = component_number
    return cuts_df


def plot_margins_center_points(sequences, n, cuts, dividing_paths=None, legend=False):
    """
    Plot margin sequences with dividing points and optionally dividing paths.
    @param sequences: list of lists with margin sequences in a form [(x,y), (y,z), ...]
    @param n: number of margins to plot
    @param cuts: dictionary with points assigned to cuts
    @param dividing_paths: dictionary of dividing paths
    @param legend: boolean, if include legend, default = False
    @return: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    color = iter(cm.rainbow(np.linspace(0, 1, n + 1)))
    for i, sequence in enumerate(sequences[:n + 1]):
        if legend:
            collection_lines = mc.LineCollection(sequence, linestyle="-",
                                                 label="Margin" + str(i),
                                                 colors=next(color), zorder=1)
        else:
            collection_lines = mc.LineCollection(sequence, linestyle="-",
                                                 colors=next(color), zorder=1)
        ax.add_collection(collection_lines)
    for value in cuts.values():
        starting_point = value[0]
        ax.scatter(x=starting_point[0], y=starting_point[1], color="green", s=40, marker="d", zorder=2)
    if dividing_paths:
        for key in dividing_paths:
            xs = [i[0] for i in dividing_paths[key]]
            ys = [i[1] for i in dividing_paths[key]]
            ax.scatter(x=xs, y=ys, color="black", s=5)
    plt.plot()
    plt.grid(True)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Starting points for cuts")
    plt.show()
    return fig


def cuts_interval(
        sequences,
        n=9,
        unit=1000,
        mode="distance",
):
    """
    Given sequences of margins, it divides the most inner margin based on mode and unit threshold,
    producing starting points for cutting analysis.
    @param sequences: list of sequences of margins
    @param n: number of margins to analyse
    @param unit: unit interval for either count or distance threshold
    @param mode: count or distance mode for dividing the most inner layer,
    [TO DO] implement density mode
    @return: dictionary with assignment of points to cuts
    """
    sequences_n = sequences[:n + 1]
    print(f"Number of layers analysed = {len(sequences_n)}")
    inner_layer_seq = [tuple(i[0]) for i in sequences_n[-1]]

    unit_counter, cut_number, current_ind = 0, 0, 0
    cuts, cut_layer_to_len = defaultdict(list), {}

    while current_ind != len(inner_layer_seq) - 1:
        while unit_counter <= unit:
            point_x = inner_layer_seq[current_ind]
            if current_ind < len(inner_layer_seq) - 1:
                point_y = inner_layer_seq[current_ind + 1]

                if mode == "distance":
                    unit_counter += distance.euclidean(point_x, point_y)
                elif mode == "count":
                    unit_counter += 1

                if point_x not in cuts[cut_number]:  # duplicated points
                    cuts[cut_number].append(point_x)

                current_ind += 1
                if current_ind == len(inner_layer_seq) - 1:  # last point
                    cuts[cut_number].append(inner_layer_seq[current_ind])
            else:
                break
        cut_layer_to_len[(cut_number, n)] = unit_counter
        cut_number += 1
        unit_counter = 0

    return cuts, cut_layer_to_len

def divide_cuts(
        df,
        cuts,
        n=9,
):
    """
    Compute splitting points for margin sequences.
    @param df: cutting dataframe
    @param cuts: dictionary with initial assignment of points to cuts
    @param n: number of margins to analyse
    @return: dictionary with dividing points for each layer: {layer: [(x,y), ...]}
    """
    starting_points = [v[0] for v in cuts.values()]
    print(f"Number of starting points = {len(starting_points)}")

    layer_trees, layer_array = {}, {}
    for layer_number in range(0, n):
        X_df = df[df["layer_number"] == layer_number]
        X_array = np.array([[x, y] for x, y in zip(X_df["nucleus.x"], X_df["nucleus.y"])])
        layer_trees[layer_number] = scipy.spatial.cKDTree(X_array)
        layer_array[layer_number] = X_array

    dividing_paths = defaultdict(list)
    for i, start in enumerate(starting_points):
        start_point = np.array(start)
        for layer in range(n - 1, -1, -1):
            ds, inds = layer_trees[layer].query(start_point, 2)
            neigh_pos = layer_array[layer][inds[0]]
            start_point = np.array(neigh_pos)
            dividing_paths[layer].append(tuple(neigh_pos))
    return dividing_paths


def assign_immune_tree(
        cuts_df,
        pos_to_cut,
        new_dist=100
):
    """
    Assign immune cells to cuts using cKDtrees queries.
    @param cuts_df: cutting dataframe
    @param pos_to_cut: dictionary with points assignment to cuts
    @return: a modified dataframe with immune points assigned to cut and layers
    """
    pos_to_layer = {(x, y): l for x, y, l in zip(cuts_df["nucleus.x"], cuts_df["nucleus.y"], cuts_df["layer_number"])}
    pos_to_layer = defaultdict(lambda: -2, pos_to_layer)

    X_df = cuts_df[(cuts_df["cut_number"] != -1) & (cuts_df["is_immune"] == False)]
    X_df = np.array([[x, y] for x, y in zip(X_df["nucleus.x"], X_df["nucleus.y"])])

    not_assigned = cuts_df[(cuts_df["cut_number"] == -1) & (cuts_df["is_immune"] == True)]
    immune_list = [[x, y] for x, y in zip(not_assigned["nucleus.x"], not_assigned["nucleus.y"])]

    ctree = scipy.spatial.cKDTree(X_df)
    for point, ind in zip(immune_list, not_assigned.index):
        point_immune = np.array(point)
        inds = ctree.query_ball_point(point_immune, new_dist, return_sorted=True)
        if inds:
            neigh_pos = X_df[inds[0]]
            cuts_df.loc[ind, "cut_number"] = pos_to_cut[tuple(neigh_pos)]
            cuts_df.loc[ind, "layer_number"] = pos_to_layer[tuple(neigh_pos)]
    return cuts_df


def assign_points_to_cuts(
        dividing_paths,
        sequences,
        cuts,
        n=9):
    """
    Assigns all points from margin sequences to cuts based on dividing paths.
    Modifies cuts dictionary by adding new assignment.
    @param dividing_paths: dictionary with dividing points for each layer
    @param sequences: list of lists of margin sequences
    @param cuts: dicionary with cuts assignment
    @param n: number of margins
    @return: a modified cuts dictionary with assignment of points to layers
    """
    for layer_number in range(n-1, -1, -1):
        current_layer = [tuple(i[0]) for i in sequences[layer_number]]
        split_points = dividing_paths[layer_number]
        ctree = scipy.spatial.cKDTree(split_points)
        for point in current_layer:
            ds, inds = ctree.query(np.array(point), 2)
            if abs(inds[0] - inds[1]) == 1:
                cut_assign = min(inds)
            elif 0 in inds:
                cut_assign = max(inds)
            elif abs(inds[0] - inds[1]) != 1:
                cut_assign = inds[0]
            cuts[cut_assign].append(point)
    return cuts


def get_layer_cut_length(dividing_paths, cuts, sequences):
    cut_layer_to_len = {}
    cuts_numbers = list(cuts.keys())
    n = max(dividing_paths.keys())
    for cut_number, next_cut in zip(cuts_numbers, cuts_numbers[1:] + [cuts_numbers[0]]):
        for layer in range(n+1):
            seq = [tuple(i[0]) for i in sequences[layer]]
            start, end = dividing_paths[layer][cut_number], dividing_paths[layer][next_cut]
            start_ind, end_ind = seq.index(start), seq.index(end)
            if start_ind < end_ind:
                seq_layer_cut = seq[start_ind:end_ind]
            else:
                seq_layer_cut = seq[start_ind:] + seq[:end_ind]
            dist = 0
            for i, j in zip(seq_layer_cut[:-1], seq_layer_cut[1:]):
                dist += distance.euclidean(i, j)
            cut_layer_to_len[(cut_number, layer)] = dist
    return cut_layer_to_len


def cut_points_n_layers(
        slicing_points,
        filename,
        component_number,
        output_path=None,
        n_margins=9,  # number of margins to analyse
        unit=50,  # unit for distance or count treshold
        mode="count",  # mode
        new_dist=100,
        plot=True,
):
    """
    Final function to perform cutting algorithm.
    @param slicing_points: dictionary with 4 keys: positions, phenotypes, sequences and margin_slices
    obtained from slicing algorithm
    @param filename: patient/sample id,
    @param component_number: number of tumor islet
    @param n_margins: number of margins to cut
    @param unit: unit interval: if mode "count" it divides inner layer into parts with unit number
    of cells, if mode "distance" it divides inner layer into parts of length equal to unit
    @param mode: can be {"distance", "count" or "density"},
    @param plot: boolean, if you want to create plots with dividing paths
    @return: a tuple with 2 objects [TO DO IF NEEDED ALL 2]: pandas dataframe with cutting summary,
    dividing paths
    """
    total_start = time.perf_counter()
    margin_slices = slicing_points["margin_slices"]
    sequences = slicing_points["sequences"]

    # Create pos to mar dict from margin slices extracted from slicing
    pos_to_mar = {tuple(point): int(key) for key in margin_slices.keys()
                  for point in margin_slices[key]}
    pos_to_mar = defaultdict(lambda: -1, pos_to_mar)  # transform to default dict

    # 1. Prepare dataframe
    cuts_df = prepare_dataframe(slicing_points, pos_to_mar, filename, component_number)

    # 2. Create initial cuts
    # start = time.perf_counter()
    cuts, cut_layer_to_len = cuts_interval(sequences, unit=unit, n=n_margins, mode=mode)
    # print(f"[CUTS INTERVAL] time: {time.perf_counter() - start}")

    if plot:
        plot_margins_center_points(sequences, n_margins, cuts)

    # start = time.perf_counter()
    dividing_paths = divide_cuts(cuts_df, cuts, n=n_margins)
    # print(f"[DIVIDE CUTS] time: {time.perf_counter() - start}")

    if plot:
        plot_margins_center_points(sequences, n_margins, cuts, dividing_paths)

    # start = time.perf_counter()
    cuts_new = assign_points_to_cuts(dividing_paths, sequences, cuts, n_margins)
    # print(f"[ASSIGN POINTS TO CUTS] time: {time.perf_counter() - start}")

    # start = time.perf_counter()
    point_to_cut = {v: k for k, l in cuts_new.items() for v in l}
    point_to_cut = defaultdict(lambda: -1, point_to_cut)
    # print(f"[REVERSE DICTIONARY] time: {time.perf_counter() - start}")

    cuts_df["cut_number"] = [point_to_cut[(x, y)] for x, y in zip(cuts_df["nucleus.x"], cuts_df["nucleus.y"])]

    # start = time.perf_counter()
    cuts_df = assign_immune_tree(cuts_df, point_to_cut, new_dist=new_dist)
    # print(f"[ASSIGN IMMUNE TREE] time: {time.perf_counter() - start}")

    # start = time.perf_counter()
    new_cuts_lens = get_layer_cut_length(dividing_paths, cuts, sequences)
    cut_layer_to_len.update(new_cuts_lens)
    cut_layer_to_len = defaultdict(int, cut_layer_to_len)
    cuts_df["cut_layer_length"] = [cut_layer_to_len[(cut, layer)]
                                   for cut, layer in zip(cuts_df["cut_number"], cuts_df["layer_number"])]
    # print(f"[GET LAYER CUTS LENS] time: {time.perf_counter() - start}")
    if output_path:
        cuts_df.to_csv(output_path, index=False)

    print(f"[TOTAL] time: {time.perf_counter() - total_start}")
    return cuts_df, dividing_paths


def visualize_cuts(cuts_df, only_cuts=True):
    """
    Visualise cutting.
    @param cuts_df: dataframe summarizing the cutting algorithm
    @param only_cuts: True if you want to visualize only cuts, False if you want to plot interior
    @return: plotly.express plot of islets cuts
    """
    cuts_df = cuts_df.sort_values(by="cut_number", ascending=True)
    cuts_df["cut_number"] = cuts_df["cut_number"].astype(str)

    if only_cuts:
        fig = px.scatter(
            cuts_df[cuts_df["cut_number"] != "-1"],
            x="nucleus.x",
            y="nucleus.y",
            color="cut_number",
            hover_data=["layer_number"]
        )
    else:
        fig = px.scatter(
            cuts_df,
            x="nucleus.x",
            y="nucleus.y",
            color="cut_number",
            hover_data=["layer_number"],
            color_discrete_map={"-1": "#cccccc"},
            title="Cuts"
        )
    fig.update_traces(marker=dict(size=2, opacity=0.7))
    fig.update_layout(legend={'itemsizing': 'constant'}, template="plotly_white")
    fig.show()
    return fig

# positions, phenotypes, sequences, margin_slices
# template_dict_path = "/home/joanna/Pulpit/Magisterka/rudy_results/cuts/cutting_211_alpha100.json"
# slicing_points = json.load(open(template_dict_path))
# df, dividing_paths = cut_points_n_layers(slicing_points, filename="test", component_number=322,
#                                          output_path="/home/joanna/Pulpit/asia.csv",
#                                                unit=2000, n_margins=50,
#                                                mode="distance", plot=False)
# print(df.head())
# visualize_cuts(df)
