import seaborn as sns
from matplotlib.pyplot import gcf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def clustermap_with_clinical_colorbar(df, df_clust, label_columns, index_column="immucan_sample_id",
                                      figsize=(10, 10),
                                      legends_placements=[(0.2, 1.03), (0.4, 1.13), (0.6, 1.13), (0.6, 1.13)],
                                      ytick_size=20, xtick_size=15, title_size=35,
                                      cmap_clustermap='coolwarm', col_cluster=True, row_cluster=True,
                                      z_score=1, method="complete", xticklabels=False, yticklabels=True,
                                      palettes=["husl", "Set2", "tab10", "Set1"], main_title="",
                                      titley=1.15, cbar_pos=(0.02, 0.8, 0.05, 0.18), dendogram=True):
    all_labels = []
    all_lut = []
    if type(label_columns) == str:
        num_cols = 1
        label_columns = [label_columns]
    else:
        num_cols = len(label_columns)
        if len(label_columns) > 4:
            label_columns = label_columns[:4]
            num_cols = 4

    for i in range(len(label_columns)):
        cols1_labels = df[label_columns[i]]
        cols1_pal = sns.color_palette(palettes[i], len(df[label_columns[i]].unique()))
        cols1_lut = dict(zip(map(str, cols1_labels.unique()), cols1_pal))
        cols1_colors = pd.Series(cols1_labels).map(cols1_lut)
        cols1_colors.index = df.reset_index()[index_column]
        all_labels.append(cols1_labels)
        all_lut.append(cols1_lut)

        if i == 0:
            color_desc = pd.DataFrame(cols1_colors.copy())
        else:
            color_desc = color_desc.join(pd.DataFrame(cols1_colors))

    # create clustermap
    sns.set(font_scale=2.5)
    g = sns.clustermap(df_clust, row_cluster=row_cluster, col_cluster=col_cluster,
                       col_colors=color_desc, figsize=figsize,
                       cmap=cmap_clustermap, xticklabels=xticklabels, yticklabels=yticklabels, z_score=z_score,
                       method=method, cbar_pos=cbar_pos)
    if not dendogram:
        g.ax_col_dendrogram.set_visible(False)
        g.ax_row_dendrogram.set_visible(False)
    if yticklabels:
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=ytick_size)
    if xticklabels:
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=xtick_size)

    g.ax_col_colors.set_yticklabels(g.ax_col_colors.get_ymajorticklabels(), fontsize=ytick_size)
    g.fig.suptitle(main_title, y=titley, fontsize=title_size, weight='bold')

    # add legends
    # add 1st label
    labs = [str(c) for c in all_labels[0].unique()]
    labs.sort()
    for label in labs:
        try:
            g.ax_col_dendrogram.bar(0, 0, color=all_lut[0][label], label=label, linewidth=0)
        except KeyError:
            print(f'{label} not in values')
            continue
    l1 = g.ax_col_dendrogram.legend(title=label_columns[0], loc="center", ncol=2, frameon=True,
                                    bbox_to_anchor=legends_placements[0], bbox_transform=gcf().transFigure,
                                    fontsize=20)
    plt.setp(l1.get_title(), fontsize="24")

    # add 2nd label
    if num_cols >= 2:
        labs = [str(c) for c in all_labels[1].unique()]
        labs.sort()
        for label in labs:
            try:
                g.ax_row_dendrogram.bar(0, 0, color=all_lut[1][label], label=label, linewidth=0)
            except KeyError:
                print(f'{label} not in values')
                continue
        l2 = g.ax_row_dendrogram.legend(title=label_columns[1], loc="center", ncol=2, frameon=True,
                                        bbox_to_anchor=legends_placements[1],
                                        bbox_transform=gcf().transFigure,
                                        fontsize=20)
        plt.setp(l2.get_title(), fontsize="24")
    #         if num_cols == 2:
    #             plt.gca().add_artist(l1)

    # add 3rd label
    if num_cols >= 3:
        xx = []
        labs = [str(c) for c in all_labels[2].unique()]
        labs.sort()
        for label in labs:
            x = g.ax_row_dendrogram.bar(0, 0, color=all_lut[2][label], label=label, linewidth=0)
            xx.append(x)
        legend3 = plt.legend(xx, labs, loc="center", title=label_columns[2], ncol=2, frameon=True,
                             bbox_to_anchor=legends_placements[2], bbox_transform=gcf().transFigure,
                             fontsize=20)
        plt.setp(legend3.get_title(), fontsize="24")

    # add 4th label
    if num_cols == 4:
        yy = []
        labs = [str(c) for c in all_labels[3].unique()]
        labs.sort()
        for label in labs:
            y = g.ax_row_dendrogram.bar(0, 0, color=all_lut[3][label], label=label, linewidth=0)
            yy.append(y)

        legend4 = plt.legend(yy, labs, loc="center", title=label_columns[3], ncol=2, frameon=True,
                             bbox_to_anchor=legends_placements[3], bbox_transform=gcf().transFigure,
                             fontsize=20)
        plt.setp(legend4.get_title(), fontsize="24")

        plt.gca().add_artist(legend3)
    plt.show()
    return g
