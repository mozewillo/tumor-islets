import matplotlib.pyplot as plt
import numpy as np
def plot_matrix_heatmap(selected_matrix, cts=None, round_to=2, title=""):
    """plot matrix as heatmap with values rounded to round_to number after the decimal point"""
    fig, ax = plt.subplots()
    im = ax.imshow(selected_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(cts)), labels=cts)
    ax.set_yticks(np.arange(len(cts)), labels=cts)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(cts)):
        for j in range(len(cts)):
            text = ax.text(j, i, selected_matrix[i, j].round(round_to),
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()