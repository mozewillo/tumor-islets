import pandas as pd
import os

_PANEL_TO_MARKER_LIST = {'IF1': ['CD11c', 'CD15', 'CD163', 'CD20', 'CD3', 'CK'],
                         'IF2': ['CD8', 'CK', 'GB', 'Ki67', 'PD1', 'PDL1'],
                         'IF3': ['CD3', 'CD4', 'CD56', 'CD8', 'CK', 'FOXP3']}

def cell_type_layers(
    cell_types,
    number_of_layers,
):
    res = []
    for layer_id in range(1, number_of_layers+1):
        res += ["L" + str(layer_id) + "_" + ct for ct in cell_types]
    return res


def peel_margin(
    steep_region,
    margin_number,
    cell_types,
):
    selected_margin = steep_region[steep_region["margin_in_component_number"] == margin_number]
    cell_type_counts = selected_margin.groupby(["cell_type"]).agg({"cell_type": "size"})["cell_type"]
    for ct in cell_types:
        if ct not in cell_type_counts.keys():
            cell_type_counts[ct] = 0
    sorted_counts = {k:v for k,v in sorted(cell_type_counts.items())}
    return list(sorted_counts.values())


def sort_phenotypes_in_steep_region(
    steep_region,
    panel=None,
):
    new_phenotypes = []
    for phen in steep_region["phenotype"]:
        new = ""
        for marker in _PANEL_TO_MARKER_LIST[panel]:
            i = phen.find(marker)
            new += phen[i:i+len(marker)+1]
        new_phenotypes.append(new)
    steep_region["phenotype"] = new_phenotypes
    return steep_region
    

def peel_n_margins_from_steep_region(
    steep_region,
    steep_region_number,
    phen_dict,
    n_margins=10,
    panel=None,
):
    cell_types = sorted(list(set(phen_dict.values())))
    steep_region = steep_region.copy()
    steep_region = steep_region[steep_region["margin_in_component_number"] != -1]
    steep_region = sort_phenotypes_in_steep_region(steep_region, panel)
    steep_region["cell_type"] = [phen_dict[i] for i in steep_region["phenotype"]]
    margin_max = steep_region["margin_in_component_number"].max()
    margin_min = steep_region["margin_in_component_number"].min()
    if margin_max - margin_min < n_margins:
        return None
    row = []
    for n in range(margin_min, margin_min + n_margins):
        row.extend(peel_margin(steep_region, n, cell_types))
    res = pd.DataFrame(columns = cell_type_layers(cell_types, n_margins))
    res.loc[0] = row
    res["steep_region_number"] = steep_region_number
    return res


def summarize_steep_regions_from_component(
    component,
    component_number,
    phen_dict,
    n_margins=10,
    panel=None,
):
    cell_types = sorted(list(set(phen_dict.values())))
    steep_regions_ids = list(set(component["steep_region_number"]))
    steep_regions_df = []
    for s_id in steep_regions_ids:
        if s_id != -1:
            sel_steep_region = component[component["steep_region_number"] == s_id]
            sr_df = peel_n_margins_from_steep_region(sel_steep_region, s_id, phen_dict, 
                                                    n_margins, panel)
            if sr_df is not None:
                steep_regions_df.append(sr_df)
    if not steep_regions_df:
        return None
    component_summary = pd.concat(steep_regions_df)
    component_summary["component_number"] = component_number
    return component_summary


def summarize_steep_regions_from_patient(
    graph_path,
    phen_dict,
    n_margins=10,
    panel=None,
):
    component_paths = [graph_path + "/Components/" + i + "/" + i + "_sliced_df.csv"
                      for i in os.listdir(graph_path + "/Components/")
                      if i != "component_-1"]
    component_paths = [c for c in component_paths if os.path.exists(c)]
    summary_dfs = []
    for path in component_paths:
        component_df = pd.read_csv(path)
        if "steep_region_number" in component_df.columns:
            component_number = path.split("component_")[1][:-1]
            df = summarize_steep_regions_from_component(component_df, component_number, 
                                                        phen_dict, n_margins, panel)
            if df is not None:
                summary_dfs.append(df)
    if not summary_dfs:
        return None
    patient_summary = pd.concat(summary_dfs)
    patient_summary["filename"] = path.split("GRAPH_")[1].split("_dist")[0]
    return patient_summary

# def summarize_steep_regions_from_cohort(
#     cohort_path,
#     phen_dict,
#     n_margins
# ):
#     graphs = [cohort_path + path for path in os.listdir(cohort_path) if "GRAPH_" in path]
#     steep_regions_cohort = []
#     for graph in graphs:
#         print(graph)
#         df = summarize_steep_regions_from_patient(graph, phen_dict, n_margins)
#         steep_regions_cohort.append(df)
#     return pd.concat(steep_regions_cohort)

def summarize_steep_regions_from_cohort_select_panel(
    cohort_path,
    phen_dict,
    n_margins=10,
    panel=None,
):
    graphs = [cohort_path + path for path in os.listdir(cohort_path) if "GRAPH_" in path]
    if panel:
        graphs = [graph for graph in graphs if panel in graph]
    steep_regions_cohort = []
    for graph in graphs:
        print(graph)
        df = summarize_steep_regions_from_patient(graph, phen_dict, n_margins, panel)
        steep_regions_cohort.append(df)
    final = pd.concat(steep_regions_cohort).reset_index(drop=True)
    cols = list(final.columns)
    final = final[[cols[-1]] + cols[:-1]]
    return final
