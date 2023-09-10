import os
#import logging
from collections import defaultdict
from tumor_islets.graph.Cell import Cell
# import tumor_islets.graph.concave_hull
from tumor_islets.graph import concave_hull
import linecache
import tracemalloc
import matplotlib.pyplot as plt
#logging.basicConfig(level=logging.WARN)

def no_position_tuple():
    return -1, -1

def no_position():
    return -1

# Function to display memory usage
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

# Function to divide list into chunks
def divide_chunks(my_list, number_of_chunks, sorted_list=False):
    # looping till length l
    if sorted_list:  # it is sorted
        total = 0
        idx_from = 0
        while total < len(my_list):
            section = my_list[idx_from:len(my_list):number_of_chunks]
            total += len(section)
            idx_from += 1
            yield section
    else:
        for i in range(0, len(my_list), number_of_chunks):
            yield my_list[i:i + number_of_chunks]

# Sth for parallel processes, come back later...
def create_cells_parallel(initial_structure, indices):
    tmp_position_to_cell_mapping = dict()
    tmp_id_to_position_mapping = dict()
    for i in indices:
        cell_1 = Cell(initial_structure.loc[i], i)
        tmp_id_to_position_mapping[i] = (cell_1.x, cell_1.y)
        tmp_position_to_cell_mapping[(cell_1.x, cell_1.y)] = cell_1
    return tmp_id_to_position_mapping, tmp_position_to_cell_mapping

def _determine_margin_helper(number, boundary):
    """
    Returns:
        - margin_edge_sequence:     sequence of (n-1,n) positions on margin
        - margin_positions:         list of cell positions in the margin
        - position_to_component:    dictionary of cell_position -> component (margin) number
    Parameters:
        - number:                   number of margin
        - boundary                  object
    """
    start, last = None, None
    margin_positions, margin_edge_sequence = [], []
    position_to_component = defaultdict(list)  # dictionary cell_position -> component_number

    for px, py in zip(*boundary.xy):
        position = (px, py)
        margin_positions.append(position)
        if not start:
            start = position
        position_to_component[position] = number
        if not last:
            last = position
            continue
        margin_edge_sequence.append([last, position])
        last = position
    return margin_edge_sequence, margin_positions, position_to_component

def determine_margin_parallel(points_for_concave_hull, number, alpha=20):
    if len(points_for_concave_hull) <= 4:
        return None, None
    # concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
    try:
        concave_hull_list, edge_points = concave_hull.alpha_shape(points_for_concave_hull, alpha=alpha)
    except Exception as e:
        print("Empty edges for component: {}", number)
        edge_points = None
    if not edge_points:
        return [], [], defaultdict(list)
    elif type(concave_hull_list.boundary).__name__ == "LineString":
        return _determine_margin_helper(boundary=concave_hull_list.boundary, number=number)
    else:
        _inv_mar_seq_tmp, _inv_mar_tmp = [], []
        _position_to_component = defaultdict(list)
        for geom in concave_hull_list.geoms:
            margin_edge_sequence, margin_positions, position_to_component = _determine_margin_helper(
                boundary=geom.boundary,
                number=number
            )
            _inv_mar_seq_tmp.extend(margin_edge_sequence)
            _inv_mar_tmp.extend(margin_positions)
            _position_to_component.update(position_to_component)
        return _inv_mar_seq_tmp, _inv_mar_tmp, _position_to_component

def _parallel_helper_margin_detection(islets, alpha):
    tmp_invasive_margins_sequence = dict()
    tmp_invasive_margins = dict()
    position_to_component = defaultdict(list)

    for idx, (islet_number, points_for_concave_hull) in enumerate(islets):
        _seq, _cells, _position_to_component = determine_margin_parallel(points_for_concave_hull, islet_number,
                                                                         alpha=alpha)
        tmp_invasive_margins_sequence[islet_number] = _seq
        tmp_invasive_margins[islet_number] = _cells
        position_to_component.update(_position_to_component)

    return tmp_invasive_margins_sequence, tmp_invasive_margins, position_to_component


def _create_color_dictionary(all_phenotypes):
    no_pheno = len(all_phenotypes)
    ck_phenotypes = [v for v in all_phenotypes if "CK" in v]
    ck_phenotypes.sort(key=len)
    non_ck_phenotypes = [v for v in all_phenotypes if not "CK" in v and v != "neg"]
    no_ck_phenotypes = len(ck_phenotypes)
    no_non_ck_phenotypes = len(non_ck_phenotypes)
    greens_palette = [plt.cm.get_cmap("Greens_r", no_non_ck_phenotypes + 1)(i)
                      for i in range(no_non_ck_phenotypes + 1)]
    reds_palette = [plt.cm.get_cmap("YlOrRd_r", no_ck_phenotypes + 1)(i)
                    for i in range(no_ck_phenotypes + 1)]
    ck_colors = dict(zip(ck_phenotypes, reds_palette[:no_ck_phenotypes]))
    non_ck_colors = dict(zip(non_ck_phenotypes, greens_palette[:no_non_ck_phenotypes]))
    color_dict_phenotypes = dict(ck_colors, **non_ck_colors)
    color_dict_phenotypes["neg"] = (0, 0.7, 1, 1)
    return color_dict_phenotypes

def helper_margin_slices(points, alpha=20):
    """
    Helper function for components slicing.
    """
    edge_points = None
    points_for_concave_hull = points
    # print("Number of points to calculate the concave hull: " + str(len(points)))
    if len(points_for_concave_hull) <= 4:
        return None
    try:
        concave_hull_list, edge_points = concave_hull.alpha_shape(points_for_concave_hull, alpha=alpha)
    except:
        print("In helper_margin_slices(points, alpha): Empty edges for component: {}")
    if not edge_points:
        return [], []
    elif type(concave_hull_list.boundary).__name__ == "LineString":
        return _determine_margin_helper(boundary=concave_hull_list.boundary, number=None)[0:2]
    else:
        _inv_mar_seq_tmp = []
        _inv_mar_tmp = []
        for geom in concave_hull_list.geoms:
            margin_edge_sequence, margin_positions, position_to_component = _determine_margin_helper(
                boundary=geom.boundary,
                number=None
            )
            _inv_mar_seq_tmp.extend(margin_edge_sequence)
            _inv_mar_tmp.extend(margin_positions)
        return _inv_mar_seq_tmp, _inv_mar_tmp
