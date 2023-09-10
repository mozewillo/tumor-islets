import pandas as pd
import json
import numpy as np
from scipy.spatial import cKDTree


def prepare_df(points):
    df = pd.DataFrame({'x': np.array(points["positions"])[:, 0],
                       'y': np.array(points["positions"])[:, 1],
                       'phenotype': points["phenotypes"]})
    df.loc[:, "is_CK"] = df["phenotype"].apply(lambda x: "CK+" in x)
    df_CK = df[df["is_CK"] == True]
    df_CK = df_CK.reset_index(drop=True)
    return df, df_CK


def cluster_cKD_n(df_CK, n_clust=10):
    df_CK["sampled"] = False
    n_samples = int(df_CK.shape[0] / n_clust)
    df_CK.loc[df_CK.sample(n=n_samples).index, "sampled"] = True

    n = df_CK[df_CK["sampled"] == False][["x", "y"]].shape[0]
    print(f"Number of not sampled points = {n}")

    # Create a cKD tree
    ind_dict = {i: j for i, j in zip(range(n + 1), df_CK[df_CK["sampled"] == False][["x", "y"]].index)}
    tree = cKDTree(np.array(df_CK[df_CK["sampled"] == False][["x", "y"]]))
    sampled_df = df_CK[df_CK["sampled"] == True]

    cluster_labels = np.array([-1] * df_CK.shape[0])
    c_nb = 0

    for ind, sampled_point in zip(sampled_df.index, np.array(sampled_df[['x', 'y']])):
        neighbour_ind = tree.query(sampled_point, n_clust - 1)[1]
        cluster_labels[ind] = c_nb
        cluster_labels[[ind_dict[i] for i in neighbour_ind]] = c_nb
        c_nb += 1

    df_CK.loc[:, "cluster"] = cluster_labels

    # Assign not clustered points
    not_assigned = df_CK[df_CK["cluster"] == -1]
    tree_post = cKDTree(np.array(df_CK[df_CK["cluster"] != -1][["x", "y"]]))
    cluster_labels = list(df_CK[df_CK["cluster"] != -1]['cluster'])

    for ind, point in zip(not_assigned.index, np.array(not_assigned[["x", 'y']])):
        neighbor_ind = tree_post.query(point, 1)[1]
        c_nb = cluster_labels[neighbor_ind]
        df_CK.loc[ind, "cluster"] = c_nb

    return df_CK


def compute_and_save_centers(df_CK, output_path):
    centers = df_CK.groupby("cluster").agg({"x": "mean", 'y': "mean", 'phenotype': 'max'})
    centers['x'] = centers['x'].apply(lambda x: round(x, 4))
    centers['y'] = centers['y'].apply(lambda x: round(x, 4))
    df_CK['nucleus.x'] = df_CK["cluster"].apply(lambda x: centers.loc[x, 'x'])
    df_CK['nucleus.y'] = df_CK["cluster"].apply(lambda x: centers.loc[x, 'y'])
    df_CK.to_csv(output_path, index=False)
    return df_CK, centers


def save_new_points(centers, df, output_path):
    df_immune = df[df['is_CK'] == False]
    new_points = {
        'positions': list([x, y] for x, y in zip(centers['x'], centers['y'])) + list([x, y] for x, y in zip(df_immune['x'], df_immune['y'])),
        'phenotypes': list(centers['phenotype']) + list(df_immune['phenotype'])
    }
    with open(output_path, "w") as outfile:
        json.dump(new_points, outfile)
    return new_points


def save_centers_cKD_clustering(points_path, output_path, n_clust=10, threshold=200000):
    points = json.load(open(points_path, "r"))
    n_points = len(points['positions'])
    if n_points >= threshold:
        df, df_CK = prepare_df(points)
        df_CK_clustered = cluster_cKD_n(df_CK, n_clust=n_clust)
        df_CK_clustered_centers, centers = compute_and_save_centers(df_CK_clustered, output_path+'.csv')
        new_points = save_new_points(centers, df, output_path+'.json')
        return df_CK_clustered_centers
    else:
        with open(output_path + '.json', "w") as outfile:
            json.dump(points, outfile)
        return 0

