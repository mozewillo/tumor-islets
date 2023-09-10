import pandas as pd
import numpy as np


def calculate_marker_coactivity(data, markers, directory, filename):
    coactivity = np.zeros((len(markers), len(markers)))
    for i, m1 in enumerate(markers):
        for j, m2 in enumerate(markers):
            act = len(data.loc[(data[f'{m1}.score.normalized'] > 1)])
            if act > 0:
                coactivity[i][j] = len(
                    data.loc[(data[f'{m1}.score.normalized'] > 1) & (data[f'{m2}.score.normalized'] > 1)]) / act
            else:
                coactivity[i][j] = np.nan
    coactivity = pd.DataFrame(coactivity, columns=markers, index=markers)
    coactivity.index.name = 'marker'
    coactivity.to_csv(f'{directory}markers_coactivity_{filename}.tsv', sep='\t')
    return coactivity


def marker_coactivity_gather(files, samples, markers):
    possible_pairs = []
    samples_list = []
    for c1 in markers:
        for c2 in markers:
            possible_pairs.append(f'{c1}_{c2}')

    for fi, (f, sample_name) in enumerate(zip(files, samples)):
        cs = pd.read_csv(f, sep='\t')
        dif = [i for i in markers if i not in cs.columns[1:]]

        if len(dif) == 0:
            cs = cs[['marker'] + markers]
            samples_list.append(sample_name)
            # include only samples with expression of all markers
            for i, ct in enumerate(markers):
                v = cs.loc[cs['marker'] == ct].iloc[0][1:].to_numpy()
                if i == 0:
                    sample_vec = v
                else:
                    sample_vec = np.hstack([sample_vec, v])
            if fi == 0:
                all_samples = sample_vec
            else:
                all_samples = np.vstack([all_samples, sample_vec])
        else:
            print(f'{sample_name} not included {dif}')

    all_samples = all_samples.astype('float')
    all_samples_df = pd.DataFrame(all_samples, index=samples_list, columns=possible_pairs)
    all_samples_df.index.name = 'sample_id'
    return all_samples_df


def calculate_mean_matrix(all_sumarized, markers):
    ms = all_sumarized.mean(axis=0)
    for j in range(6, 37, 6):
        if j == 6:
            means = ms[:j].to_numpy()
        else:
            means = np.vstack([means, ms[i:j].to_numpy()])
        i = j
    means = pd.DataFrame(means, columns=markers, index=markers)
    means.index.name = 'marker'
    return means
