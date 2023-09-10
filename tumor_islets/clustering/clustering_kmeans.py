from matplotlib.pyplot import gcf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np

# K means visualisation kit for k choice
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
import logging
import warnings
from sklearn.exceptions import DataConversionWarning

np.random.seed(0)


def silhouette_kmeans(X, ks=[2, 3, 4, 5, 6]):
    """
    Silhouette Score = (b-a)/max(a,b)
    where, a= average intra-cluster distance i.e the average distance between each point within a cluster.
    Args:
        X: data for K-means clustering
        ks: list of Ks for which to calculate silhouette score
    Returns:
        Plots the silhouette score for each K proposed for K-means
    """
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    r = int(np.ceil(len(ks) / 2))
    fig, ax = plt.subplots(r, 2, figsize=(20, 8))
    for i, k in enumerate(ks):
        '''
        Create KMeans instances for different number of clusters
        '''
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q, mod])
        visualizer.fit(X)


def elbow_kmeans(X):
    """Plots Elbow curve - WCSS (Within-Cluster Sum of Square)
    the sum of the square distance between points in a cluster and the cluster centroid
    Args:
        X: data for K-means clustering
    """
    # Instantiate the clustering model and visualizer
    warnings.filterwarnings(action='ignore')

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2, 10))

    visualizer.fit(X)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure


def phen_to_cell_dict(phen_path):
    """Create a dictionary from phenotype to cell"""
    phenotype_key = pd.read_csv(phen_path)
    phenotype_to_cell = dict(zip(phenotype_key["phenotype"],
                                 phenotype_key["celltype"]))
    return phenotype_to_cell


def rename_and_groupby(df, phenotype_to_cell):
    cell_columns = sorted(list(set(phenotype_to_cell.values())))
    non_cell_columns = [c for c in df.columns
                        if c not in phenotype_to_cell.keys()]
    df["filename"] = [d.split("_#_cell")[0] for d in df["filename"]]
    df = df.rename(columns=phenotype_to_cell)
    df = df.groupby(level=0, axis=1).sum()
    df = df.reindex(columns=non_cell_columns + cell_columns)
    return df


def read_csv_IF(path, phenotype_key_path=None):
    """Read dataframe from path. If a dictionnary of phenotypes to cell types
    is given, then rename columns and groupby cell types."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.iloc[:, 1:]
    print(f"Shape: {df.shape}")
    if phenotype_key_path:
        phenotype_to_cell = phen_to_cell_dict(phenotype_key_path)
        df = rename_and_groupby(df, phenotype_to_cell)
    return df


def hellinger_distance(df, ncols=3):
    islet_size = df.iloc[:, ncols:].sum(axis=1)
    df.iloc[:, ncols:] = df.iloc[:, ncols:].div(df.iloc[:, ncols:].sum(axis=1), axis=0)
    df.iloc[:, ncols:] = np.sqrt(df.iloc[:, ncols:]).div(np.sqrt(2), axis=1)
    df["islet_size"] = np.sqrt(np.log(islet_size)).div(np.sqrt(2))
    return df


def create_clustermap(df):
    marker_cm = sns.clustermap(df.T, cmap='coolwarm', figsize=(20, 9),
                               col_cluster=True, row_cluster=False, z_score=1)

    y_ticks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    _ = marker_cm.ax_cbar.set_yticks((0.5 * y_ticks) ** 0.5)
    _ = marker_cm.ax_cbar.set_yticklabels(y_ticks, fontsize=10.0)


def visualize_centers(centers, columns):
    """
    Visualize in plotly centers of cluters:
    centers - np.array with cluster centers,
    columns - column names for clustered features.
    """
    centers = pd.DataFrame(centers)
    centers.columns = columns
    n_clust = centers.shape[0]
    centers.index = ["Cluster " + str(i) for i in range(n_clust)]
    fig = px.imshow(centers.T, color_continuous_scale="Bluered", text_auto=".2f")
    fig.layout.title = "Cluster centers for K=" + str(n_clust)
    fig.show()
    return fig

def get_most_variable_cols(df, nvars=30):
    col_vars = pd.DataFrame(df.var())
    col_vars.reset_index(inplace=True)
    col_vars.columns = ['pair', 'var']
    top_vars = list(col_vars.nlargest(nvars, 'var')['pair'])
    return top_vars

def visualize_centers_long(centers, columns, nvars=30, width=800, height=800):
    """
    Visualize  centers of cluters when many features:
    centers - np.array with cluster centers,
    columns - column names for clustered features.
    nvars - how many variables to plot - sorted by variance
    """
    centers = pd.DataFrame(centers)
    centers.columns = columns
    n_clust = centers.shape[0]
    centers.index = ["Cluster " + str(i) for i in range(n_clust)]
    most_var = get_most_variable_cols(centers, nvars=nvars)
    centers_plot = centers[most_var]
    fig = px.imshow(centers_plot.T, color_continuous_scale="Bluered", text_auto=".2f")
    fig.update_layout(autosize=False, width=width, height=height)
    fig.layout.title = "Cluster centers for K=" + str(n_clust)
    fig.show()
    return fig


def visualize_centers_long_pyplot(centers, columns, nvars=30, figsize=(20,8)):
    """
    Visualize  centers of cluters when many features:
    centers - np.array with cluster centers,
    columns - column names for clustered features.
    nvars - how many variables to plot - sorted by variance
    """
    centers = pd.DataFrame(centers)
    centers.columns = columns
    n_clust = centers.shape[0]
    centers.index = ["Cluster " + str(i) for i in range(n_clust)]
    most_var = get_most_variable_cols(centers, nvars=nvars)
    centers_plot = centers[most_var]
    marker_cm = sns.clustermap(centers_plot,
                           cmap='coolwarm',
                           figsize=figsize,
                           col_cluster = True,
                           row_cluster = False,
                           z_score=None,
                           method="complete"
                          )
    return marker_cm
def cluster_and_visualize_Kmeans(n_clust, df, df_patient):
    """
    Perform K-means clustering:
    - n_clust - number of clusters,
    - df - dataframe to cluster,
    - df_patient - original dataframe to append a column with cluster number.
    Returns a modified df_patient and centers of clusters.
    """
    clustering = KMeans(n_clusters=n_clust, random_state=5).fit(df)
    df_patient["cluster"] = clustering.labels_
    return df_patient, clustering.cluster_centers_


def patient_level_clustering(df_clust, id_column="sample_id"):
    """
    Summarize each patient as a collection of features from different clusters.
    """
    sorted_filenames = sorted(list(set(df_clust[id_column])))
    n_clust = len(set(df_clust["cluster"]))
    grouped = df_clust.groupby([id_column, "cluster"]).agg({"cluster": "count"})
    df = pd.DataFrame(index=sorted_filenames,
                      columns=["Cluster_" + str(i) for i in range(n_clust)])
    df = df.fillna(0)
    for key, value in dict(grouped["cluster"]).items():
        df.loc[key[0], "Cluster_" + str(key[1])] = value
    df = df.div(df.sum(axis=1), axis=0).T
    return df


def summarize_patients(df_clust, df_cohort, color_bar_column, legend_title=None, title=None):
    labels = df_cohort[color_bar_column]
    lut = dict(zip(set(labels), sns.hls_palette(len(set(labels)), h=.3)))
    col_colors = df_cohort[color_bar_column].map(lut)
    col_colors.index = df_cohort["immucan_sample_id"]
    g = sns.clustermap(df_clust, row_cluster=False, col_colors=col_colors, figsize=(10, 10))
    if title is not None:
        g.fig.suptitle(title, fontsize=16, x=0.2, y=1.01)
    for label in labels.unique():
        g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0);
    if not legend_title:
        legend_title = color_bar_column
    l1 = g.ax_col_dendrogram.legend(title=legend_title, loc="center", ncol=3,
                                    bbox_to_anchor=(0.8, 1.0),
                                    bbox_transform=gcf().transFigure)
    plt.show()
    return g
