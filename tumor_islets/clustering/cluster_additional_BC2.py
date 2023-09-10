import tumor_islets.clustering.clustering_kmeans as clustering_IF
import pandas as pd
import os
import seaborn as sns
from matplotlib.pyplot import gcf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger

def cluster_and_check_subtypes(cluster_cols, K, subtypes, subtypes_na, hellinger_df, prefix, pdf_path=None):
    """Create 3 plots"""
    df_clust, centers = clustering_IF.cluster_and_visualize_Kmeans(K, hellinger_df.iloc[:, cluster_cols[0]:cluster_cols[1]], df)
    df_clust["sample_id"] = df_clust["filename"].apply(lambda x: prefix + x.split("-IF")[0])
    cen = clustering_IF.visualize_centers(centers, hellinger_df.iloc[:, cluster_cols[0]:cluster_cols[1]].columns)
    
    patient = clustering_IF.patient_level_clustering(df_clust)
    keep_patients = [p for p in subtypes_na["immucan_sample_id"] if p in patient.columns]
    print(f"How many to keep = {len(keep_patients)}")
    patients_na = patient[keep_patients]

    figs = []
    for col in subtypes.columns[1:4]:
        figs.append(clustering_IF.summarize_patients(patients_na, subtypes_na, col))
    
    if pdf_path is not None:
        cen.write_image("temp1.pdf")
        p = PdfPages("temp2.pdf")
        for fig in figs:
            fig.savefig(p, format="pdf")
        p.close()
    
        merger = PdfMerger()
        for pdf in ["temp1.pdf", "temp2.pdf"]:
            merger.append(pdf)
        merger.write(pdf_path)
        merger.close()
        os.remove("temp1.pdf")
        os.remove("temp2.pdf")
    
    return df_clust, centers, figs


# saved - viridis, seismic, cubehelix

def clustermap_subtypes_colorbar(df_clust, df_cohort):
    burstein_labels = df_cohort["burstein"]
    burstein_lut = dict(zip(set(burstein_labels), sns.color_palette("hls", len(set(burstein_labels)))))
    burstein_colors = df_cohort["burstein"].map(burstein_lut)
    burstein_colors.index = df_cohort["immucan_sample_id"]
    
    time_labels = df_cohort["TIME"]
    time_lut = dict(zip(set(time_labels), sns.color_palette("turbo", len(set(time_labels)))))
    time_colors = df_cohort["TIME"].map(time_lut)
    time_colors.index = df_cohort["immucan_sample_id"]
    
    tnbc_labels = df_cohort["TNBCsubtype"]
    tnbc_lut = dict(zip(set(tnbc_labels), sns.color_palette("cubehelix", len(set(tnbc_labels)))))
    tnbc_colors = df_cohort["TNBCsubtype"].map(tnbc_lut)
    tnbc_colors.index = df_cohort["immucan_sample_id"]
    
    survival_labels = df_cohort["[POST_Sample]_TRTSG_PCR_1"]
    survival_lut = dict(zip(set(survival_labels), sns.color_palette("CMRmap", len(set(survival_labels)))))
    survival_colors = df_cohort["[POST_Sample]_TRTSG_PCR_1"].map(survival_lut)
    survival_colors.index = df_cohort["immucan_sample_id"]

    time_burstein_colors = pd.DataFrame(tnbc_colors).join(pd.DataFrame(time_colors)).join(pd.DataFrame(burstein_colors)).join(pd.DataFrame(survival_colors))
    
    g = sns.clustermap(df_clust, row_cluster=False, col_colors=time_burstein_colors, figsize=(10, 10),
    #                   cmap="plasma",
                      )

    for label in tnbc_labels.unique():
        g.ax_col_dendrogram.bar(0, 0, color=tnbc_lut[label], label=label, linewidth=0.98);
    l1 = g.ax_col_dendrogram.legend(title="TNBC subtype", loc="right", ncol=3, 
                                    bbox_to_anchor=(0.6, 1.05), 
                                    bbox_transform=gcf().transFigure)

    for label in time_labels.unique():
        g.ax_row_dendrogram.bar(0, 0, color=time_lut[label], label=label, linewidth=0);
    l2 = g.ax_row_dendrogram.legend(title="TIME subtype", loc="right", ncol=3, 
                                    bbox_to_anchor=(1.15, 1.06), 
                                    bbox_transform=gcf().transFigure)
    xx = []
    for label in burstein_labels.unique():
        x = g.ax_row_dendrogram.bar(0, 0, color=burstein_lut[label], label=label, linewidth=0)
        xx.append(x)
#     l3 = g.ax_row_dendrogram.legend(title='Burstein subtype', loc="right", ncol=3, 
#                                     bbox_to_anchor=(1.15, 1.0), 
#                                     bbox_transform=gcf().transFigure)
    
    legend3 = plt.legend(xx, burstein_labels.unique(), loc="right", title='Burstein subtype', ncol=4,
                         bbox_to_anchor=(1.15, 1.0), bbox_transform=gcf().transFigure)

    yy= []
    for label in survival_labels.unique():
        y = g.ax_row_dendrogram.bar(0, 0, color=survival_lut[label], label=label, linewidth=0)
        yy.append(y)
        
    legend4 = plt.legend(yy, survival_labels.unique(), loc="right", title="[POST_Sample]_TRTSG_PCR_1", ncol=4,
                         bbox_to_anchor=(1.15, 0.9), bbox_transform=gcf().transFigure)
    
    plt.gca().add_artist(legend3)

    plt.show()
    return g


def cluster_and_color_subtypes(cluster_cols, subtypes_na, hellinger_df, prefix,  K, pdf_path=None):
    """Save 2 plots to one pdf"""
    df_clust, centers = clustering_IF.cluster_and_visualize_Kmeans(K, hellinger_df.iloc[:, cluster_cols[0]:cluster_cols[1]], df)
    df_clust["sample_id"] = df_clust["filename"].apply(lambda x: prefix + x.split("-IF")[0])
    cen = clustering_IF.visualize_centers(centers, hellinger_df.iloc[:, cluster_cols[0]:cluster_cols[1]].columns)
    
    patient = clustering_IF.patient_level_clustering(df_clust)
    keep_patients = [p for p in subtypes_na["immucan_sample_id"] if p in patient.columns]
    print(f"How many to keep = {len(keep_patients)}")
    patients_na = patient[keep_patients]

    fig = clustermap_subtypes_colorbar(patients_na, subtypes_na)
      
    if pdf_path is not None:
        cen.write_image("temp1.pdf")
        fig.savefig("temp2.pdf", format="pdf")
        
        merger = PdfMerger()
        for pdf in ["temp1.pdf", "temp2.pdf"]:
            merger.append(pdf)
        merger.write(pdf_path)
        merger.close()
        os.remove("temp1.pdf")
        os.remove("temp2.pdf")
    
    return df_clust, centers, fig