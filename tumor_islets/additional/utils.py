import re

##### Panels markers sets #####
panels_desc = {
    '1': ('CD11c', 'CD15', 'CD163', 'CD20', 'CD3', 'CK'),
    '2': ('CD8', 'CK', 'GB', 'Ki67', 'PD1', 'PDL1'),
    '3': ('CD3', 'CD4', 'CD56', 'CD8', 'CK', 'FOXP3')}


def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'


###### data.frame preprocessing and filling in inputting missing columns #####
def roi_tumor_tissue(df, rule=lambda x: True):
    if not 'in.ROI.tumor_tissue' in df.columns:
        df['in.ROI.tumor_tissue'] = [rule(row) for row in df.iterrows()]
    return df


def scores(df, panel_number):
    ## check if column names contain "MARKER.score" and "MARKER.score.normalized" 
    if not any(['score.normalized' in col_name for col_name in df.columns]):
        # non-standard columns names: only marker names without '.score/.score.normalized' suffixes
        print("scores")
        print(panel_number)
        print(df.columns)
        try:
            new_col_names = [m + ".score.normalized" for m in panels_desc[panel_number]]
            print(new_col_names)
            df.rename(columns=dict(zip(panels_desc[panel_number], new_col_names)))
        except Exception as e:
            print(
                "[ERROR] Trying to infer the scores for markers because score.normalized was not found but an error occured.")
            print("Accepted keys:")
            print(panels_desc[panel_number])
            raise e

    # TODO If there were no score.normalized fields
    # should we normalize the values in the dataset?

    return df

def phenotypes(df, panel_number):
    # if there is no phenotype in the data derive it 
    if not "phenotype" in df.columns:
        df['phenotype'] = [''.join([marker + "+" if row[marker + ".score.normalized"] > 1 else marker + '-'
                                    for marker in panels_desc[panel_number]]) for row in df.iterrows]
    # if there is phenotype make sure it is sorted properly 
    else:
        pattern = '[+-]'.join(panels_desc[panel_number]) + "[+-]$"
        if not bool(re.search(pattern, df['phenotype'][0])):
            sorted_phenotypes = []
            for index, row in df.iterrows():
                markers_extracted = [re.search(marker + '[+-]', row['phenotype']).group() for marker in
                                     panels_desc[panel_number]]
                sorted_phenotype = ''.join(sorted(markers_extracted))
                sorted_phenotypes.append(sorted_phenotype)
            df['phenotype'] = sorted_phenotypes
    return df
