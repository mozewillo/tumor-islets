import pandas as pd
import glob
import os


def map_to_celltype(df, mapping_file):
    mapping = pd.read_csv(mapping_file)
    phs = df['phenotype'].unique()
    phs.sort()
    marker_order = [m for p in phs[-1].split('+') for m in p.split('-')]
    original = []
    reordered = []
    for ph in mapping['phenotype'].unique():
        ph_str = ''
        for m in marker_order:
            if m != '':
                ph_str += m + ph.split(m)[-1][0]
        original.append(ph)
        reordered.append(ph_str)
    ph_dict = {'phenotype': original, 'phenotype_reordered': reordered}
    ph_dict = pd.DataFrame.from_dict(ph_dict)
    mapping = pd.merge(mapping, ph_dict, on='phenotype')
    mapping = mapping[['celltype', 'phenotype_reordered']]
    mapping.columns = ['celltype', 'phenotype']
    df = pd.merge(df, mapping, on='phenotype', how='left')
    allct = list(mapping['celltype'].unique())
    return df, allct


def create_sample_summary(input_tsv_files_dir, cohort_name, save=True, output_path='./', mapping_files_dict=None):
    """ Create summaries of marker activities % in all cells and with the regard on ROI if available
        -- providing mapping files in form of dir key=panel_name value=file_with_mapping
    """
    sample_summary = {'filename': [], 'immucan_sample_id': [], 'immucan_patient_id': [], 'IF_panel': [],
                      'properties2': [],
                      'number_of_cells': [], 'in.ROI.tumor_tissue.percent': [], 'in.ROI.necrosis.percent': [],
                      'TLS.num': [], 'in.ROI.adipose_tissue.percent': [], 'tissue.type.tumor.percent': []}

    for i, f in enumerate(glob.glob(input_tsv_files_dir + '*properties_*')):
        try:
            data = pd.read_csv(f, sep='\t')
        except:
            print(f'Cannot open file: {f}')
            continue
        properties2 = False
        filename = f.split('/')[-1]
        sample_id = filename.split('_#')[0]
        if_panel = sample_id.split('-')[-2]
        sample_id = sample_id.split('-IF')[0]
        patient_id = sample_id.split('-FIXT')[0]
        cell_num = len(data)

        if len(glob.glob(f'{input_tsv_files_dir}{sample_id}*properties2_*')) == 1:
            f2 = glob.glob(f'{input_tsv_files_dir}{sample_id}*properties2_*')[0]
            d2 = pd.read_csv(f2, sep='\t')
            data = pd.merge(data, d2, on=['cell.ID', 'nucleus.x', 'nucleus.y'])
            properties2 = True
            if len(data) != cell_num:
                print(f'Unequal number of cells in properties 2: {f2}, skipped.')
                properties2 = False
                data = pd.read_csv(f, sep='\t')

        sample_summary['filename'].append(filename)
        sample_summary['immucan_sample_id'].append(sample_id)
        sample_summary['immucan_patient_id'].append(patient_id)
        sample_summary['number_of_cells'].append(cell_num)
        sample_summary['properties2'].append(properties2)
        sample_summary['IF_panel'].append(if_panel)
        if cell_num == 0:
            print(f"No cells???: {f}")
        marker_cols = list(set([c for c in data.columns if '.normalized' in c]))

        for j, m in enumerate(marker_cols):
            m_name = m.split('score')[0] + 'positive.percent'
            if m_name not in sample_summary.keys():
                sample_summary[m_name] = [None] * (len(sample_summary['filename']) - 1)
            sample_summary[m_name].append(len(data.loc[data[m] > 1]) / cell_num)

        # map celltypes
        if mapping_files_dict is not None:
            if if_panel in mapping_files_dict.keys():
                data, allct = map_to_celltype(data, mapping_file=mapping_files_dict[if_panel])
            for ct in allct:
                c_name = f'{ct}.all.ct.percent'
                if c_name not in sample_summary.keys():
                    sample_summary[c_name] = [None] * (len(sample_summary['filename']) - 1)
                sample_summary[c_name].append(len(data.loc[(data['celltype'] == ct)]) / len(data))

        if 'in.ROI.tumor_tissue' in data.columns:
            sample_summary['in.ROI.tumor_tissue.percent'].append(
                len(data.loc[data['in.ROI.tumor_tissue'] == True]) / cell_num)
            for j, m in enumerate(marker_cols):
                m_name_2 = m.split('score')[0] + 'in.tumor.percent'
                m_name_2o = m.split('score')[0] + 'out.tumor.percent'
                if m_name_2 not in sample_summary.keys():
                    sample_summary[m_name_2] = [None] * (len(sample_summary['filename']) - 1)
                    sample_summary[m_name_2o] = [None] * (len(sample_summary['filename']) - 1)
                if len(data.loc[data['in.ROI.tumor_tissue'] == True]) > 0:
                    sample_summary[m_name_2].append(
                        len(data.loc[(data[m] > 1) & (data['in.ROI.tumor_tissue'] == True)]) / len(
                            data.loc[data['in.ROI.tumor_tissue'] == True]))
                if len(data.loc[data['in.ROI.tumor_tissue'] == False]) > 0:
                    sample_summary[m_name_2o].append(
                        len(data.loc[(data[m] > 1) & (data['in.ROI.tumor_tissue'] == False)]) / len(
                            data.loc[data['in.ROI.tumor_tissue'] == False]))
            if 'celltype' in data.columns:
                for ct in allct:
                    c_name = f'{ct}.in.tumor.ct.percent'
                    c_name2 = f'{ct}.out.tumor.ct.percent'
                    if c_name not in sample_summary.keys():
                        sample_summary[c_name] = [None] * (len(sample_summary['filename']) - 1)
                        sample_summary[c_name2] = [None] * (len(sample_summary['filename']) - 1)
                    if len(data.loc[data['in.ROI.tumor_tissue'] == True]) > 0:
                        sample_summary[c_name].append(
                            len(data.loc[(data['celltype'] == ct) & (data['in.ROI.tumor_tissue'] == True)]) / len(
                                data.loc[data['in.ROI.tumor_tissue'] == True]))
                    if len(data.loc[data['in.ROI.tumor_tissue'] == False]) > 0:
                        sample_summary[c_name2].append(len(data.loc[(data['celltype'] == ct) &
                                                                    (data['in.ROI.tumor_tissue'] == False)]) / len(
                            data.loc[data['in.ROI.tumor_tissue'] == False]))

        if 'in.ROI.necrosis' in data.columns:
            sample_summary['in.ROI.necrosis.percent'].append(len(data['in.ROI.necrosis']))

        if 'TLS.ID' in data.columns:
            sample_summary['TLS.num'].append(len(data['TLS.ID'].unique()))

        if 'in.ROI.adipose_tissue' in data.columns:
            sample_summary['in.ROI.adipose_tissue.percent'].append(
                len(data.loc[data['in.ROI.adipose_tissue'] == True]) / cell_num)

        if 'tissue.type' in data.columns:
            sample_summary['tissue.type.tumor.percent'].append(len(data.loc[data['tissue.type'] == 'tumor']) / cell_num)

        for k in sample_summary.keys():
            if len(sample_summary[k]) < len(sample_summary['filename']):
                sample_summary[k].append(None)

    df = pd.DataFrame.from_dict(sample_summary)
    if save:
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(f'{output_path}/summary_{cohort_name}.tsv', sep='\t')

    return df


def merge_info(summary, sample_col='immucan_sample_id', panel_col='IF_panel',
               basic_cols=['immucan_sample_id', 'immucan_patient_id'],
               drop_cols=['Unnamed: 0', 'immucan_patient_id', 'IF_panel', 'properties2']):
    """Merge info from different panels per sample
        Input: summary per file
        Output: summary per sample"""

    for i, p in enumerate(summary[panel_col].unique()):
        ps = summary.loc[summary[panel_col] == p]
        ps.drop_duplicates(sample_col, inplace=True)
        print('Panel :', p)
        if i == 0:
            dcol = [d for d in drop_cols if d not in basic_cols]
            ps.drop(dcol, axis=1, inplace=True)
            ps.columns = [f'{p}_{c}' if c not in basic_cols else c for c in ps.columns]
            ps[p] = True
            alls = ps
        else:
            ps.drop(drop_cols, axis=1, inplace=True)
            ps.columns = [f'{p}_{c}' if c != sample_col else c for c in ps.columns]
            ps[p] = True
            alls = pd.merge(alls, ps, on=sample_col, how='left')
    alls = alls.dropna(axis=1, how='all')
    return alls


def select_cols_and_samples(alls, sample_col='immucan_sample_id',
                            tumor_ROI='', panel=['IF1', 'IF2', 'IF3'],
                            celltypes=False):
    """Select columns from summary
        - celltypes if True select only celltype columns
        - list of panels to select
        - tumor_ROI select based on tumor tissue ROI region options: 'in', 'out', 'both'
    """
    if type(panel) == str:
        panel = [panel]
    if not celltypes:
        if tumor_ROI == 'in':
            cols_suffix = '.in.tumor.percent'
        elif tumor_ROI == 'out':
            cols_suffix = '.out.tumor.percent'
        elif tumor_ROI == 'both':
            cols_suffix = '.tumor.percent'
        else:
            cols_suffix = '.positive.percent'

    if celltypes:
        if tumor_ROI == 'in':
            cols_suffix = '.in.tumor.ct.percent'
        elif tumor_ROI == 'out':
            cols_suffix = '.out.tumor.ct.percent'
        elif tumor_ROI == 'both':
            cols_suffix = '.tumor.ct.percent'
        else:
            cols_suffix = '.all.ct.percent'

    selected_cols = [c for c in alls.columns if c.endswith(cols_suffix)]

    selected_cols = [c for c in selected_cols if c.split('_')[0] in panel]

    selected = alls[[sample_col] + selected_cols]
    selected.dropna(inplace=True)

    return selected

