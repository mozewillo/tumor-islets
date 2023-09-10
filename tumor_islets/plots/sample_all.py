import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib
import pandas as pd
import glob


def size_colormap():
    col_dict = {0: "yellow",
                1: "red",
                2: "orange",
                3: "blue",
                4: "cornflowerblue"}

    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

    labels = np.array(['Tiny (<29 cells)', 'Small (30-100 cells)', 'Medicore (101-250 cells)',
                       'Large (251-500 cells)', 'Front (>500 cells)'])
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    print(norm_bins)
    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
    return cm, norm, labels


def panel_colormaps(panel='IF1'):
    if panel == 'IF1':
        col_dict = {0: "yellow",
                    1: "hotpink",
                    2: "green",
                    3: "orange",
                    4: "blue",
                    5: "seashell"}

        labels = ['CD11c', 'CD15', 'CD163', 'CD20', 'CD3', 'CK']

    if panel == 'IF2':
        col_dict = {0: "red",
                    1: "seashell",
                    2: "lime",
                    3: "dodgerblue",
                    4: "violet",
                    5: "purple"}

        labels = ['CD8', 'CK', 'GB', 'Ki67', 'PD1', 'PDL1']

    if panel == 'IF3':
        col_dict = {0: "blue",
                    1: "cyan",
                    2: "seagreen",
                    3: "red",
                    4: "seashell",
                    5: "magenta"}

        labels = ['CD3', 'CD4', 'CD56', 'CD8', 'CK', 'FOXP3']

    marker_dict = dict([(k, v) for v, k in enumerate(labels)])

    return col_dict, labels, marker_dict


def marker_colormap(panel):
    col_dict, labels, marker_dict = panel_colormaps(panel)
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    print(norm_bins)
    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
    return cm, norm, labels, marker_dict


def visualise_components(cells, sample_name, column_name='component_number', out_name=None, ax=None, size_thr_down=40,
                         size_thr_up=2000, color_name=None, cmap_comp='Accent', norm=None):
    if color_name is None:
        if 'in.ROI.tumor_tissue' not in cells.columns:
            if 'tissue.type' in cells.columns:
                cells['tissue.type'] = cells['tissue.type'].map({'stroma': 0, 'tumor': 1})
                color_name = 'tissue.type'
            else:
                color_name = 'phenotype'
                print('no color_column')
        else:
            color_name = 'in.ROI.tumor_tissue'

    tls = cells.loc[~cells[column_name].isnull()]
    if out_name is None:
        tls['Component'] = 1
        out_name = 'Component'

    if column_name == 'component_number':
        tls = tls.loc[tls['component_number'] != -1]
    tls = tls.loc[~tls[out_name].isnull()]

    if size_thr_down is not None or size_thr_up is not None:
        sizes = pd.DataFrame([(k, v) for k, v in cells[column_name].value_counts().items()],
                             columns=[column_name, 'size'])
        if size_thr_down is not None: sizes = sizes.loc[sizes['size'] > size_thr_down]
        if size_thr_up is not None:  sizes = sizes.loc[sizes['size'] < size_thr_up]
        tls = tls.loc[tls[column_name].isin(list(sizes[column_name].unique()))]

    print(f'Number of components {len(tls[column_name].unique())}')
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    plt.style.use('ggplot')

    scatter1 = ax.scatter(x=cells["nucleus.x"], y=cells["nucleus.y"], s=3, c=cells[color_name], alpha=0.05, cmap='Set2')
    scatter2 = ax.scatter(x=tls["nucleus.x"], y=tls["nucleus.y"], s=3, c=tls[out_name], cmap=cmap_comp, norm=norm)
    legend1 = ax.legend(*scatter1.legend_elements(),
                        loc="lower left", title='in.ROI.tumor_tissue')
    for lh in legend1.legendHandles:
        lh.set_alpha(1)

    ax.add_artist(legend1)
    handles, labels = scatter2.legend_elements()
    print(labels)
    legend2 = ax.legend(handles, labels,
                        loc="lower left", bbox_to_anchor=(0, 0.1, 0, 0))
    ax.set_title(f'Detected components for {sample_name}')
#     return fig


def visualise_tls_id(cells, sample_name, ax=None):
    if 'in.ROI.tumor_tissue' not in cells.columns:
        if 'tissue.type' in cells.columns:
            cells['tissue.type'] = cells['tissue.type'].map({'stroma': 0, 'tumor': 1})
            color_name = 'tissue.type'
        else:
            color_name = None
    else:
        color_name = 'in.ROI.tumor_tissue'
    cells.loc[~cells['TLS.ID'].isnull(), 'TLS'] = 1
    tls = cells.loc[~cells['TLS.ID'].isnull()]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    plt.style.use('ggplot')

    scatter1 = ax.scatter(x=cells["nucleus.x"], y=cells["nucleus.y"], s=3, c=cells[color_name], alpha=0.05, cmap='Set2')
    scatter2 = ax.scatter(x=tls["nucleus.x"], y=tls["nucleus.y"], s=3, c=tls['TLS'], cmap='hsv')
    legend1 = ax.legend(*scatter1.legend_elements(),
                        loc="lower left", title='in.ROI.tumor_tissue')
    for lh in legend1.legendHandles:
        lh.set_alpha(1)

    ax.add_artist(legend1)
    handles, labels = scatter2.legend_elements()
    legend2 = ax.legend(handles, ['TLS'],
                        loc="lower left", bbox_to_anchor=(0, 0.1, 0, 0))
    plt.title(f'TLS from properties2 for {sample_name}')


def visualise_marker_expression(cells, marker_name, sample_name, ax=None, cm='hsv'):
    if 'in.ROI.tumor_tissue' not in cells.columns:
        if 'tissue.type' in cells.columns:
            cells['tissue.type'] = cells['tissue.type'].map({'stroma': 0, 'tumor': 1})
            color_name = 'tissue.type'
        else:
            color_name = None
    else:
        color_name = 'in.ROI.tumor_tissue'
    marker_column = f'{marker_name}.score.normalized'
    cells.loc[cells[marker_column] > 1, marker_name] = 1
    tls = cells.loc[~cells[marker_name].isnull()]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    plt.style.use('ggplot')

    if color_name is not None:
        scatter1 = ax.scatter(x=cells["nucleus.x"], y=cells["nucleus.y"], s=3, c=cells[color_name], alpha=0.05,
                              cmap='Set2')
    else:
        scatter1 = ax.scatter(x=cells["nucleus.x"], y=cells["nucleus.y"], s=3, alpha=0.05, cmap='Set2')

    scatter2 = ax.scatter(x=tls["nucleus.x"], y=tls["nucleus.y"], s=3, c=tls[marker_name], cmap=cm)
    if color_name is not None:
        legend1 = ax.legend(*scatter1.legend_elements(),
                            loc="lower left", title=color_name)
        for lh in legend1.legendHandles:
            lh.set_alpha(1)

        ax.add_artist(legend1)
    handles, labels = scatter2.legend_elements()
    legend2 = ax.legend(handles, [marker_name],
                        loc="lower left", bbox_to_anchor=(0, 0.1, 0, 0))
    plt.title(f'{marker_name} for {sample_name}')


def visualise_num_column(cells, col_name, sample_name, title, ax=None, panel='IF1', cm='magma'):
    if 'in.ROI.tumor_tissue' not in cells.columns:
        if 'tissue.type' in cells.columns:
            cells['tissue.type'] = cells['tissue.type'].map({'stroma': 0, 'tumor': 1})
            color_name = 'tissue.type'
        else:
            color_name = None
    else:
        color_name = 'in.ROI.tumor_tissue'

    tls = cells.loc[~cells[col_name].isnull()]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    plt.style.use('ggplot')

    if color_name is not None:
        scatter1 = ax.scatter(x=cells["nucleus.x"], y=cells["nucleus.y"], s=1, c=cells[color_name], alpha=0.05,
                              cmap='Set2')
    else:
        scatter1 = ax.scatter(x=cells["nucleus.x"], y=cells["nucleus.y"], s=1, alpha=0.05, cmap='Set2')

    scatter2 = ax.scatter(x=tls["nucleus.x"], y=tls["nucleus.y"], s=1, c=tls[col_name], alpha=0.15, cmap=cm)

    if color_name is not None:
        legend1 = ax.legend(*scatter1.legend_elements(),
                            loc="lower left", title=color_name)

        for lh in legend1.legendHandles:
            lh.set_alpha(1)

        ax.add_artist(legend1)
        cbar = plt.colorbar(scatter2, shrink=0.5)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel(title, rotation=270)
        cbar.set_alpha(1)
        cbar.draw_all()

    plt.title(f'{title} for {sample_name}')
