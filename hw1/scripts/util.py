import matplotlib.pyplot as plt
import numpy as np


def visualize(x_arr, labels_true, labels_pred, ws):
    unique_labels = np.unique(labels_true)
    unique_colors = dict([(label, c) for label, c in zip(
        unique_labels, [[0.8, 0., 0.], [0., 0., 0.8]])])
    plt.figure(figsize=(9, 9))

    if ws[1] == 0:
        plt.plot([x_arr[:, 0].min(), x_arr[:, 0].max()], ws[0] / ws[2])
    elif ws[2] == 0:
        plt.plot(ws[0] / ws[1], [x_arr[:, 1].min(), x_arr[:, 1].max()])
    else:
        mins, maxs = x_arr.min(axis=0), x_arr.max(axis=0)
        pts = [[mins[0], -mins[0] * ws[1] / ws[2] - ws[0] / ws[2]],
               [maxs[0], -maxs[0] * ws[1] / ws[2] - ws[0] / ws[2]],
               [-mins[1] * ws[2] / ws[1] - ws[0] / ws[1], mins[1]],
               [-maxs[1] * ws[2] / ws[1] - ws[0] / ws[1], maxs[1]]]
        pts = [(x, y) for x, y in pts
               if mins[0] <= x <= maxs[0] and mins[1] <= y <= maxs[1]]
        x, y = list(zip(*pts))
        plt.plot(x, y, c=(0.75, 0.75, 0.75), linestyle="--")

    colors_inner = [unique_colors[l_t] for l_t in labels_true]
    colors_outer = [unique_colors[l_f] for l_f in labels_pred]
    plt.scatter(x_arr[:, 0], x_arr[:, 1],
                c=colors_inner, edgecolors=colors_outer)
    plt.show()
